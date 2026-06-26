#!/usr/bin/env python3
"""
Lightning Foam Detector - Streams global lightning data and monitors for spacetime foam disturbances
Connects to Blitzortung WebSocket and MQTT feeds, processes data, and detects anomalies
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import aiohttp
import websockets
import paho.mqtt.client as mqtt
from scipy import signal as sp_signal
import pandas as pd
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the detector"""
    # Output directory (will be created if doesn't exist)
    OUTPUT_DIR = Path.home() / "lightning_foam_data"
    
    # Blitzortung WebSocket servers
    WEBSOCKET_SERVERS = [
        "wss://ws1.blitzortung.org",
        "wss://ws2.blitzortung.org",
        "wss://ws3.blitzortung.org",
        "wss://ws4.blitzortung.org"
    ]
    
    # MQTT Configuration
    MQTT_BROKER = "tcp://mqtt.blitzortung.org"
    MQTT_PORT = 1883
    MQTT_TOPICS = [
        "blitzortung/lightning/#",
        "blitzortung/raw/#"
    ]
    
    # Data collection parameters
    MAX_STRIKES_IN_MEMORY = 10000
    SAVE_INTERVAL_SECONDS = 300  # Save data every 5 minutes
    RECONNECT_DELAY_SECONDS = 5
    
    # Foam detection parameters
    FOAM_FREQUENCIES = {
        'fundamental': 2.005e-3,  # 2.005 mHz (1 AU light-crossing)
        'lunar_tidal': 0.022e-3,  # 0.022 mHz (M2 lunar tide)
        'schumann_1': 7.83,       # Schumann fundamental
        'schumann_2': 14.3,       # Schumann harmonic
        'schumann_3': 20.8        # Schumann harmonic
    }
    
    # Detection thresholds
    POWER_THRESHOLD = 3.0  # Sigma threshold for anomaly detection
    CLUSTER_THRESHOLD = 5  # Minimum strikes in cluster
    TIME_WINDOW_SECONDS = 60  # Window for temporal analysis
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LightningStrike:
    """Represents a single lightning strike"""
    timestamp: float
    latitude: float
    longitude: float
    power: float  # Estimated current in kA
    polarity: int  # -1 or 1
    region: str
    source: str  # 'websocket' or 'mqtt'
    
    def to_dict(self):
        return asdict(self)

@dataclass
class FoamEvent:
    """Represents a detected spacetime foam disturbance"""
    event_id: str
    start_time: float
    end_time: float
    center_lat: float
    center_lon: float
    total_energy: float
    frequency_components: Dict[str, float]
    anomaly_score: float
    strikes_involved: List[LightningStrike]
    
    def to_dict(self):
        data = asdict(self)
        data['strikes_involved'] = [s.to_dict() for s in self.strikes_involved]
        return data

# ============================================================================
# DATA STORAGE MANAGER
# ============================================================================

class DataStorage:
    """Manages data persistence and recovery"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.ensure_directories()
        self.current_file = None
        self.file_start_time = None
        
    def ensure_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.output_dir,
            self.output_dir / "raw_strikes",
            self.output_dir / "foam_events",
            self.output_dir / "analysis",
            self.output_dir / "logs",
            self.output_dir / "recovery"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Save configuration
        config_path = self.output_dir / "config.json"
        config_data = {
            'output_dir': str(self.output_dir),
            'created': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_current_file(self):
        """Get or create current data file"""
        now = datetime.utcnow()
        if (self.current_file is None or 
            self.file_start_time is None or
            (now - self.file_start_time) > timedelta(hours=1)):
            
            # Close previous file if open
            if self.current_file:
                self.current_file.close()
                
            # Create new file
            filename = f"strikes_{now.strftime('%Y%m%d_%H%M%S')}.jsonl"
            filepath = self.output_dir / "raw_strikes" / filename
            self.current_file = open(filepath, 'a')
            self.file_start_time = now
            self.save_recovery_state()
            
        return self.current_file
    
    def save_strike(self, strike: LightningStrike):
        """Save a lightning strike to file"""
        file = self.get_current_file()
        data = strike.to_dict()
        data['saved_at'] = datetime.utcnow().isoformat()
        file.write(json.dumps(data) + '\n')
        file.flush()
    
    def save_foam_event(self, event: FoamEvent):
        """Save a foam event detection"""
        filename = f"event_{event.event_id}.json"
        filepath = self.output_dir / "foam_events" / filename
        with open(filepath, 'w') as f:
            json.dump(event.to_dict(), f, indent=2)
    
    def save_recovery_state(self):
        """Save recovery state for restart"""
        state = {
            'last_file': self.current_file.name if self.current_file else None,
            'last_save': datetime.utcnow().isoformat(),
            'strike_count': self.get_strike_count()
        }
        state_path = self.output_dir / "recovery" / "last_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f)
    
    def get_strike_count(self) -> int:
        """Get approximate strike count from files"""
        count = 0
        pattern = self.output_dir / "raw_strikes" / "strikes_*.jsonl"
        for filepath in pattern.parent.glob(pattern.name):
            try:
                with open(filepath, 'r') as f:
                    count += sum(1 for _ in f)
            except:
                pass
        return count
    
    def load_recovery_state(self):
        """Load recovery state and return last file if exists"""
        state_path = self.output_dir / "recovery" / "last_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
        return None

# ============================================================================
# FOAM DETECTION ENGINE
# ============================================================================

class FoamDetector:
    """Detects spacetime foam disturbances from lightning patterns"""
    
    def __init__(self, config: Config):
        self.config = config
        self.strike_buffer = deque(maxlen=config.MAX_STRIKES_IN_MEMORY)
        self.event_counter = 0
        
        # Statistics for anomaly detection
        self.power_stats = {'mean': 0, 'std': 1, 'count': 0}
        self.time_stats = {'mean_interval': 0, 'std_interval': 1}
        
    def add_strike(self, strike: LightningStrike):
        """Add a strike and check for foam events"""
        self.strike_buffer.append(strike)
        self.update_statistics(strike)
        
        # Check for temporal clustering
        events = self.check_temporal_clustering()
        
        # Check for spectral anomalies
        events.extend(self.check_spectral_anomalies())
        
        # Check for spatial patterns
        events.extend(self.check_spatial_patterns())
        
        return events
    
    def update_statistics(self, strike: LightningStrike):
        """Update running statistics for anomaly detection"""
        # Update power statistics (Welford's algorithm)
        self.power_stats['count'] += 1
        delta = strike.power - self.power_stats['mean']
        self.power_stats['mean'] += delta / self.power_stats['count']
        delta2 = strike.power - self.power_stats['mean']
        self.power_stats['std'] += delta * delta2
        
        # Update time statistics if we have previous strike
        if len(self.strike_buffer) > 1:
            prev_strike = self.strike_buffer[-2]
            interval = strike.timestamp - prev_strike.timestamp
            # Simple exponential moving average for time intervals
            if self.time_stats['mean_interval'] == 0:
                self.time_stats['mean_interval'] = interval
                self.time_stats['std_interval'] = interval * 0.1
            else:
                alpha = 0.1
                self.time_stats['mean_interval'] = (
                    alpha * interval + 
                    (1 - alpha) * self.time_stats['mean_interval']
                )
                self.time_stats['std_interval'] = (
                    alpha * abs(interval - self.time_stats['mean_interval']) +
                    (1 - alpha) * self.time_stats['std_interval']
                )
    
    def check_temporal_clustering(self) -> List[FoamEvent]:
        """Check for unusual temporal clustering of strikes"""
        if len(self.strike_buffer) < self.config.CLUSTER_THRESHOLD * 2:
            return []
        
        events = []
        window_seconds = self.config.TIME_WINDOW_SECONDS
        
        # Look for clusters in recent strikes
        recent_strikes = list(self.strike_buffer)[-100:]  # Last 100 strikes
        if len(recent_strikes) < self.config.CLUSTER_THRESHOLD:
            return []
        
        # Find clusters in time
        clusters = []
        current_cluster = []
        
        for i, strike in enumerate(recent_strikes):
            if not current_cluster:
                current_cluster.append(strike)
            else:
                time_diff = strike.timestamp - current_cluster[-1].timestamp
                if time_diff < window_seconds:
                    current_cluster.append(strike)
                else:
                    if len(current_cluster) >= self.config.CLUSTER_THRESHOLD:
                        clusters.append(current_cluster.copy())
                    current_cluster = [strike]
        
        # Check last cluster
        if len(current_cluster) >= self.config.CLUSTER_THRESHOLD:
            clusters.append(current_cluster)
        
        # Analyze each cluster for foam signatures
        for cluster in clusters:
            if self.analyze_cluster_for_foam(cluster):
                event = self.create_foam_event(cluster, "temporal_cluster")
                events.append(event)
        
        return events
    
    def check_spectral_anomalies(self) -> List[FoamEvent]:
        """Check for spectral signatures in strike timing"""
        if len(self.strike_buffer) < 100:
            return []
        
        # Get recent strike timestamps
        recent_strikes = list(self.strike_buffer)[-100:]
        timestamps = [s.timestamp for s in recent_strikes]
        
        # Convert to time series
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = end_time - start_time
        
        if duration < 10:  # Need sufficient time for spectral analysis
            return []
        
        # Create time series with 1-second bins
        time_series = np.zeros(int(duration))
        for ts in timestamps:
            idx = int(ts - start_time)
            if 0 <= idx < len(time_series):
                time_series[idx] += 1
        
        # Compute power spectral density
        freqs, psd = sp_signal.welch(time_series, fs=1.0, nperseg=min(64, len(time_series)))
        
        # Check for peaks at foam frequencies
        events = []
        for name, target_freq in self.config.FOAM_FREQUENCIES.items():
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - target_freq))
            if idx < len(psd):
                power = psd[idx]
                
                # Check if significantly above noise floor
                noise_floor = np.median(psd)
                if power > noise_floor * self.config.POWER_THRESHOLD:
                    # Get strikes in the relevant time window
                    relevant_strikes = [
                        s for s in recent_strikes 
                        if start_time <= s.timestamp <= end_time
                    ]
                    
                    if len(relevant_strikes) >= 5:
                        event = self.create_foam_event(
                            relevant_strikes, 
                            f"spectral_{name}",
                            frequency_data={name: power/noise_floor}
                        )
                        events.append(event)
        
        return events
    
    def check_spatial_patterns(self) -> List[FoamEvent]:
        """Check for unusual spatial patterns in strikes"""
        if len(self.strike_buffer) < 20:
            return []
        
        recent_strikes = list(self.strike_buffer)[-50:]
        
        # Calculate spatial centroid
        lats = [s.latitude for s in recent_strikes]
        lons = [s.longitude for s in recent_strikes]
        
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # Calculate dispersion
        distances = [
            self.haversine(s.latitude, s.longitude, center_lat, center_lon)
            for s in recent_strikes
        ]
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Check for unusually tight clustering
        if mean_distance < 100 and std_distance < 50:  # km
            if self.analyze_cluster_for_foam(recent_strikes):
                event = self.create_foam_event(recent_strikes, "spatial_cluster")
                return [event]
        
        return []
    
    def analyze_cluster_for_foam(self, cluster: List[LightningStrike]) -> bool:
        """Analyze if a cluster shows foam-like signatures"""
        if len(cluster) < 5:
            return False
        
        # Check power distribution
        powers = [s.power for s in cluster]
        power_mean = np.mean(powers)
        power_std = np.std(powers)
        
        # Check timing intervals
        timestamps = sorted([s.timestamp for s in cluster])
        intervals = np.diff(timestamps)
        
        # Look for harmonic intervals (suggestive of resonance)
        if len(intervals) >= 3:
            interval_mean = np.mean(intervals)
            interval_cv = np.std(intervals) / interval_mean if interval_mean > 0 else 1
            
            # Low coefficient of variation suggests regular timing
            if interval_cv < 0.3:
                return True
        
        # Check for high-power concentration
        if power_mean > self.power_stats['mean'] + 2 * (self.power_stats['std'] ** 0.5):
            return True
        
        return False
    
    def create_foam_event(self, strikes: List[LightningStrike], 
                         event_type: str,
                         frequency_data: Optional[Dict] = None) -> FoamEvent:
        """Create a FoamEvent from detected strikes"""
        self.event_counter += 1
        
        # Calculate event parameters
        timestamps = [s.timestamp for s in strikes]
        lats = [s.latitude for s in strikes]
        lons = [s.longitude for s in strikes]
        powers = [s.power for s in strikes]
        
        event_id = f"{int(time.time())}_{self.event_counter:06d}"
        
        # Calculate frequency components if not provided
        if frequency_data is None:
            frequency_data = {}
            # Simple FFT on timestamps
            if len(timestamps) >= 10:
                intervals = np.diff(sorted(timestamps))
                if len(intervals) > 1:
                    freqs = 1.0 / intervals
                    for name, target_freq in self.config.FOAM_FREQUENCIES.items():
                        if target_freq < 1:  # Low frequencies only
                            closest = freqs[np.argmin(np.abs(freqs - target_freq))]
                            frequency_data[name] = float(closest)
        
        return FoamEvent(
            event_id=event_id,
            start_time=min(timestamps),
            end_time=max(timestamps),
            center_lat=float(np.mean(lats)),
            center_lon=float(np.mean(lons)),
            total_energy=float(np.sum(powers)),
            frequency_components=frequency_data,
            anomaly_score=self.calculate_anomaly_score(strikes),
            strikes_involved=strikes
        )
    
   def calculate_anomaly_score(self, strikes: List[LightningStrike]) -> float:
        """Calculate anomaly score for a set of strikes"""
        if not strikes:
            return 0.0
        
        scores = []
        
        # Power anomaly
        powers = [s.power for s in strikes]
        power_zscore = (np.mean(powers) - self.power_stats['mean']) / max(1e-6, (self.power_stats['std'] ** 0.5))
        scores.append(abs(power_zscore))
        
        # Temporal clustering
        if len(strikes) > 1:
            timestamps = sorted([s.timestamp for s in strikes])
            intervals = np.diff(timestamps)
            if len(intervals) > 0:
                interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1
                # Low CV (regular intervals) gets higher score
                scores.append(1.0 / max(interval_cv, 0.1))
        
        # Spatial clustering
        lats = [s.latitude for s in strikes]
        lons = [s.longitude for s in strikes]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        distances = [
            self.haversine(s.latitude, s.longitude, center_lat, center_lon)
            for s in strikes
        ]
        spatial_score = 100.0 / (np.mean(distances) + 1)  # Higher for tighter clusters
        scores.append(spatial_score)
        
        return float(np.mean(scores))
    
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate great-circle distance between two points in km"""
        R = 6371.0  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

# ============================================================================
# DATA STREAM HANDLERS
# ============================================================================

class BlitzortungWebSocketHandler:
    """Handles WebSocket connection to Blitzortung"""
    
    def __init__(self, storage: DataStorage, detector: FoamDetector):
        self.storage = storage
        self.detector = detector
        self.ws_url = None
        self.websocket = None
        self.running = False
        self.logger = logging.getLogger("WebSocket")
    
    async def connect(self):
        """Connect to Blitzortung WebSocket"""
        for server in Config.WEBSOCKET_SERVERS:
            try:
                self.logger.info(f"Connecting to {server}")
                self.websocket = await websockets.connect(server)
                self.ws_url = server
                
                # Send initialization message
                init_msg = {"a": 111}  # Standard Blitzortung init message
                await self.websocket.send(json.dumps(init_msg))
                self.logger.info("Connected and initialized")
                return True
            except Exception as e:
                self.logger.error(f"Failed to connect to {server}: {e}")
                continue
        
        self.logger.error("Could not connect to any WebSocket server")
        return False
    
    async def listen(self):
        """Listen for and process incoming messages"""
        self.running = True
        reconnect_attempts = 0
        
        while self.running:
            try:
                if not self.websocket:
                    if not await self.connect():
                        await asyncio.sleep(Config.RECONNECT_DELAY_SECONDS)
                        reconnect_attempts += 1
                        if reconnect_attempts > 10:
                            self.logger.error("Too many reconnection failures")
                            break
                        continue
                
                message = await self.websocket.recv()
                await self.process_message(message)
                reconnect_attempts = 0  # Reset on successful message
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed, reconnecting...")
                self.websocket = None
                await asyncio.sleep(Config.RECONNECT_DELAY_SECONDS)
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket listener: {e}")
                await asyncio.sleep(1)
    
    async def process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            # Blitzortung uses a custom binary format, but for this example
            # we'll simulate decoding. In reality, you'd need their decoder.
            
            # Simulated decoded strike data
            strike_data = self.simulate_strike_decoder(message)
            
            if strike_data:
                strike = LightningStrike(
                    timestamp=time.time(),
                    latitude=strike_data['lat'],
                    longitude=strike_data['lon'],
                    power=strike_data.get('power', 10.0),  # kA estimate
                    polarity=strike_data.get('polarity', 1),
                    region=strike_data.get('region', 'unknown'),
                    source='websocket'
                )
                
                # Save and analyze
                self.storage.save_strike(strike)
                events = self.detector.add_strike(strike)
                
                for event in events:
                    self.storage.save_foam_event(event)
                    self.logger.info(f"Foam event detected: {event.event_id} "
                                   f"(score: {event.anomaly_score:.2f})")
                    
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def simulate_strike_decoder(self, message):
        """Simulate Blitzortung message decoding"""
        # In reality, you'd use the actual Blitzortung decoder
        # This is a placeholder that returns simulated data
        import random
        
        # Simulate occasional strikes
        if random.random() < 0.1:  # 10% chance of simulated strike
            return {
                'lat': random.uniform(-90, 90),
                'lon': random.uniform(-180, 180),
                'power': random.uniform(5, 100),
                'polarity': random.choice([-1, 1]),
                'region': random.choice(['NA', 'EU', 'AS', 'SA', 'AF', 'OC'])
            }
        return None
    
    def stop(self):
        """Stop the WebSocket listener"""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())

class MQTTHandler:
    """Handles MQTT connection to Blitzortung"""
    
    def __init__(self, storage: DataStorage, detector: FoamDetector):
        self.storage = storage
        self.detector = detector
        self.client = mqtt.Client()
        self.logger = logging.getLogger("MQTT")
        
        # Set up callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            self.logger.info("MQTT Connected successfully")
            # Subscribe to topics
            for topic in Config.MQTT_TOPICS:
                client.subscribe(topic)
                self.logger.info(f"Subscribed to {topic}")
        else:
            self.logger.error(f"MQTT Connection failed with code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback for MQTT messages"""
        try:
            # Decode MQTT message (Blitzortung uses JSON)
            data = json.loads(msg.payload.decode())
            
            # Extract strike information
            strike = self.parse_mqtt_message(data, msg.topic)
            if strike:
                self.storage.save_strike(strike)
                events = self.detector.add_strike(strike)
                
                for event in events:
                    self.storage.save_foam_event(event)
                    self.logger.info(f"Foam event from MQTT: {event.event_id}")
                    
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        if rc != 0:
            self.logger.warning(f"MQTT Unexpected disconnection (rc={rc})")
    
    def parse_mqtt_message(self, data: dict, topic: str) -> Optional[LightningStrike]:
        """Parse MQTT message into LightningStrike"""
        try:
            # Parse based on topic structure
            if 'lightning' in topic or 'raw' in topic:
                # Extract coordinates (Blitzortung uses geohash or lat/lon)
                lat = data.get('lat', data.get('latitude', 0))
                lon = data.get('lon', data.get('longitude', 0))
                
                # Extract power/current estimate
                power = data.get('power', data.get('current', 10.0))
                
                # Extract timestamp
                timestamp = data.get('timestamp', time.time())
                
                # Extract polarity
                polarity = data.get('polarity', 1)
                if isinstance(polarity, str):
                    polarity = 1 if polarity.upper() in ['POSITIVE', '+'] else -1
                
                # Determine region from coordinates
                region = self.get_region(lat, lon)
                
                return LightningStrike(
                    timestamp=timestamp,
                    latitude=lat,
                    longitude=lon,
                    power=float(power),
                    polarity=int(polarity),
                    region=region,
                    source='mqtt'
                )
        except Exception as e:
            self.logger.error(f"Error parsing MQTT data: {e}")
        return None
    
    @staticmethod
    def get_region(lat: float, lon: float) -> str:
        """Get region code from coordinates"""
        if -180 <= lon < -30:
            if lat > 0:
                return 'NA'  # North America
            else:
                return 'SA'  # South America
        elif -30 <= lon < 60:
            if lat > 0:
                return 'EU'  # Europe/Africa
            else:
                return 'AF'  # Africa
        elif 60 <= lon < 150:
            return 'AS'  # Asia
        else:
            return 'OC'  # Oceania
    
    def start(self):
        """Start MQTT client"""
        try:
            self.logger.info(f"Connecting to MQTT broker: {Config.MQTT_BROKER}")
            self.client.connect(Config.MQTT_BROKER.split('://')[1], Config.MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            self.logger.error(f"Failed to start MQTT: {e}")
    
    def stop(self):
        """Stop MQTT client"""
        self.client.loop_stop()
        self.client.disconnect()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class LightningFoamDetector:
    """Main application class"""
    
    def __init__(self):
        self.config = Config()
        self.storage = DataStorage(self.config.OUTPUT_DIR)
        self.detector = FoamDetector(self.config)
        self.ws_handler = None
        self.mqtt_handler = None
        self.running = False
        self.logger = self.setup_logging()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format=self.config.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.OUTPUT_DIR / "logs" / "detector.log")
            ]
        )
        return logging.getLogger("LightningFoamDetector")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    async def run_websocket(self):
        """Run WebSocket listener"""
        self.ws_handler = BlitzortungWebSocketHandler(self.storage, self.detector)
        await self.ws_handler.listen()
    
    def run_mqtt(self):
        """Run MQTT listener"""
        self.mqtt_handler = MQTTHandler(self.storage, self.detector)
        self.mqtt_handler.start()
    
    async def periodic_save(self):
        """Periodically save state and statistics"""
        while self.running:
            await asyncio.sleep(self.config.SAVE_INTERVAL_SECONDS)
            self.storage.save_recovery_state()
            
            # Log statistics
            strike_count = self.storage.get_strike_count()
            self.logger.info(f"Statistics: {strike_count} total strikes, "
                          f"{self.detector.event_counter} foam events detected")
            
            # Save detector state
            self.save_detector_state()
    
    def save_detector_state(self):
        """Save detector state for recovery"""
        state = {
            'event_counter': self.detector.event_counter,
            'power_stats': self.detector.power_stats,
            'time_stats': self.detector.time_stats,
            'saved_at': datetime.utcnow().isoformat()
        }
        state_path = self.config.OUTPUT_DIR / "recovery" / "detector_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_detector_state(self):
        """Load detector state from recovery"""
        state_path = self.config.OUTPUT_DIR / "recovery" / "detector_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.detector.event_counter = state.get('event_counter', 0)
                self.detector.power_stats = state.get('power_stats', 
                    {'mean': 0, 'std': 1, 'count': 0})
                self.detector.time_stats = state.get('time_stats',
                    {'mean_interval': 0, 'std_interval': 1})
                self.logger.info("Loaded detector state from recovery")
            except Exception as e:
                self.logger.error(f"Error loading detector state: {e}")
    
    async def run(self):
        """Main application loop"""
        self.logger.info("Starting Lightning Foam Detector")
        self.logger.info(f"Output directory: {self.config.OUTPUT_DIR}")
        
        # Check for recovery
        recovery_state = self.storage.load_recovery_state()
        if recovery_state:
            self.logger.info(f"Recovering from previous state: {recovery_state}")
            self.load_detector_state()
        
        self.running = True
        
        # Start MQTT in background thread
        self.run_mqtt()
        
        # Start WebSocket and periodic tasks
        websocket_task = asyncio.create_task(self.run_websocket())
        save_task = asyncio.create_task(self.periodic_save())
        
        # Monitor tasks
        monitor_task = asyncio.create_task(self.monitor_tasks([websocket_task, save_task]))
        
        try:
            await monitor_task
        except asyncio.CancelledError:
            self.logger.info("Application cancelled")
        finally:
            self.stop()
    
    async def monitor_tasks(self, tasks):
        """Monitor tasks and restart if needed"""
        while self.running:
            for task in tasks:
                if task.done():
                    if task.exception():
                        self.logger.error(f"Task failed: {task.exception()}")
                        # Restart the task
                        if task == tasks[0]:  # WebSocket task
                            tasks[0] = asyncio.create_task(self.run_websocket())
                        elif task == tasks[1]:  # Save task
                            tasks[1] = asyncio.create_task(self.periodic_save())
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def stop(self):
        """Stop the application"""
        self.running = False
        self.logger.info("Shutting down...")
        
        if self.ws_handler:
            self.ws_handler.stop()
        
        if self.mqtt_handler:
            self.mqtt_handler.stop()
        
        # Final save
        self.storage.save_recovery_state()
        self.save_detector_state()
        
        self.logger.info("Shutdown complete")

# ============================================================================
# DATA ANALYSIS AND VISUALIZATION
# ============================================================================

class FoamAnalyzer:
    """Analyzes collected data for foam patterns"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.logger = logging.getLogger("FoamAnalyzer")
    
    def load_strikes(self, days: int = 7) -> pd.DataFrame:
        """Load strikes from the last N days"""
        strikes = []
        pattern = self.data_dir / "raw_strikes" / "strikes_*.jsonl"
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        for filepath in pattern.parent.glob(pattern.name):
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get('timestamp', 0) > cutoff_time:
                            strikes.append(data)
            except Exception as e:
                self.logger.error(f"Error reading {filepath}: {e}")
        
        return pd.DataFrame(strikes)
    
    def analyze_foam_patterns(self, df: pd.DataFrame):
        """Analyze strike patterns for foam signatures"""
        if df.empty:
            return {"error": "No data available"}
        
        results = {
            "total_strikes": len(df),
            "time_range": {
                "start": datetime.fromtimestamp(df['timestamp'].min()).isoformat(),
                "end": datetime.fromtimestamp(df['timestamp'].max()).isoformat(),
                "duration_hours": (df['timestamp'].max() - df['timestamp'].min()) / 3600
            },
            "spatial_distribution": {
                "lat_range": [float(df['latitude'].min()), float(df['latitude'].max())],
                "lon_range": [float(df['longitude'].min()), float(df['longitude'].max())],
                "centroid": [float(df['latitude'].mean()), float(df['longitude'].mean())]
            },
            "power_statistics": {
                "mean": float(df['power'].mean()),
                "std": float(df['power'].std()),
                "min": float(df['power'].min()),
                "max": float(df['power'].max()),
                "median": float(df['power'].median())
            },
            "temporal_patterns": {},
            "spectral_analysis": {},
            "foam_candidates": []
        }
        
        # Temporal analysis
        df_sorted = df.sort_values('timestamp')
        time_diffs = np.diff(df_sorted['timestamp'])
        
        if len(time_diffs) > 0:
            results["temporal_patterns"]["mean_interval_s"] = float(np.mean(time_diffs))
            results["temporal_patterns"]["std_interval_s"] = float(np.std(time_diffs))
            results["temporal_patterns"]["cv_interval"] = float(
                np.std(time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
            )
            
            # Check for regular intervals (low CV suggests possible resonance)
            if results["temporal_patterns"]["cv_interval"] < 0.3:
                results["temporal_patterns"]["regular_intervals"] = True
                results["temporal_patterns"]["estimated_period_s"] = float(np.mean(time_diffs))
            else:
                results["temporal_patterns"]["regular_intervals"] = False
        
        # Spectral analysis
        if len(df) >= 50:
            # Create time series for spectral analysis
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            duration = end_time - start_time
            
            if duration > 3600:  # At least 1 hour of data
                # Bin strikes into 1-second intervals
                time_bins = np.arange(start_time, end_time, 1)
                strike_counts, _ = np.histogram(df['timestamp'], bins=time_bins)
                
                # Compute power spectral density
                freqs, psd = sp_signal.welch(strike_counts, fs=1.0, nperseg=min(256, len(strike_counts)))
                
                # Look for peaks at foam frequencies
                foam_freqs = {
                    '2_mHz': 2.005e-3,
                    'lunar_tidal': 0.022e-3,
                    'schumann_7.8': 7.83,
                    'schumann_14.3': 14.3,
                    'schumann_20.8': 20.8
                }
                
                for name, target_freq in foam_freqs.items():
                    if target_freq < freqs[-1]:  # Only if within Nyquist
                        idx = np.argmin(np.abs(freqs - target_freq))
                        if idx < len(psd):
                            power = psd[idx]
                            noise_floor = np.median(psd)
                            snr = power / noise_floor if noise_floor > 0 else 0
                            
                            results["spectral_analysis"][name] = {
                                "frequency_hz": float(target_freq),
                                "power": float(power),
                                "snr": float(snr),
                                "significant": snr > 3.0
                            }
        
        # Spatial clustering analysis
        from sklearn.cluster import DBSCAN
        
        if len(df) >= 20:
            coords = df[['latitude', 'longitude']].values
            
            # Use DBSCAN for spatial clustering
            # Convert to radians for haversine distance
            coords_rad = np.radians(coords)
            
            # Approximate: 1 degree ≈ 111 km at equator
            # Use 50 km epsilon
            eps_rad = 50 / 6371.0  # Convert km to radians
            
            db = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine')
            labels = db.fit_predict(coords_rad)
            
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise label
            
            results["spatial_clusters"] = {
                "n_clusters": len(unique_labels),
                "cluster_sizes": [],
                "cluster_centers": []
            }
            
            for label in unique_labels:
                cluster_points = coords[labels == label]
                results["spatial_clusters"]["cluster_sizes"].append(len(cluster_points))
                results["spatial_clusters"]["cluster_centers"].append(
                    [float(cluster_points[:, 0].mean()), float(cluster_points[:, 1].mean())]
                )
        
        # Identify foam event candidates
        if 'source' in df.columns:
            # Look for high-power clusters in short time windows
            df['time_bin'] = (df['timestamp'] // 60) * 60  # 1-minute bins
            
            for time_bin, group in df.groupby('time_bin'):
                if len(group) >= 5:  # At least 5 strikes in 1 minute
                    avg_power = group['power'].mean()
                    if avg_power > results["power_statistics"]["mean"] + 2 * results["power_statistics"]["std"]:
                        candidate = {
                            "time": datetime.fromtimestamp(time_bin).isoformat(),
                            "duration_s": 60,
                            "n_strikes": len(group),
                            "avg_power": float(avg_power),
                            "center_lat": float(group['latitude'].mean()),
                            "center_lon": float(group['longitude'].mean()),
                            "z_score": float((avg_power - results["power_statistics"]["mean"]) / 
                                           max(1e-6, results["power_statistics"]["std"]))
                        }
                        results["foam_candidates"].append(candidate)
        
        return results
    
    def generate_report(self, analysis_results: dict, output_path: Path):
        """Generate HTML report of analysis"""
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Lightning Foam Analysis Report', fontsize=16)
        
        # 1. Time series of strikes
        ax = axes[0, 0]
        df = self.load_strikes(1)  # Last 24 hours
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            ax.plot(df['datetime'], df['power'], 'b.', alpha=0.5, markersize=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Power (kA)')
            ax.set_title('Strike Power vs Time')
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.grid(True, alpha=0.3)
        
        # 2. Spatial distribution
        ax = axes[0, 1]
        if not df.empty:
            scatter = ax.scatter(df['longitude'], df['latitude'], 
                               c=df['power'], s=10, alpha=0.6, 
                               cmap='hot', vmin=0, vmax=100)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Spatial Distribution (color = power)')
            plt.colorbar(scatter, ax=ax, label='Power (kA)')
            ax.grid(True, alpha=0.3)
        
        # 3. Power histogram
        ax = axes[1, 0]
        if not df.empty:
            ax.hist(df['power'], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Power (kA)')
            ax.set_ylabel('Count')
            ax.set_title('Power Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean: {analysis_results['power_statistics']['mean']:.1f} kA\n"
            stats_text += f"Std: {analysis_results['power_statistics']['std']:.1f} kA\n"
            stats_text += f"Max: {analysis_results['power_statistics']['max']:.1f} kA"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Temporal intervals
        ax = axes[1, 1]
        if not df.empty and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            intervals = np.diff(df_sorted['timestamp'])
            ax.hist(intervals, bins=50, alpha=0.7, edgecolor='black', log=True)
            ax.set_xlabel('Time between strikes (s)')
            ax.set_ylabel('Count (log)')
            ax.set_title('Temporal Interval Distribution')
            ax.grid(True, alpha=0.3)
            
            if 'temporal_patterns' in analysis_results:
                tp = analysis_results['temporal_patterns']
                if 'mean_interval_s' in tp:
                    stats_text = f"Mean: {tp['mean_interval_s']:.1f} s\n"
                    stats_text += f"Std: {tp['std_interval_s']:.1f} s\n"
                    if tp.get('regular_intervals', False):
                        stats_text += "Regular intervals detected!"
                        ax.text(0.05, 0.95, "⚠️ REGULAR\nINTERVALS", 
                               transform=ax.transAxes, color='red',
                               verticalalignment='top', fontweight='bold')
        
        # 5. Spectral analysis results
        ax = axes[2, 0]
        if 'spectral_analysis' in analysis_results and analysis_results['spectral_analysis']:
            freqs = []
            snrs = []
            labels = []
            
            for name, data in analysis_results['spectral_analysis'].items():
                if data.get('significant', False):
                    freqs.append(data['frequency_hz'])
                    snrs.append(data['snr'])
                    labels.append(name)
            
            if freqs:
                bars = ax.bar(range(len(freqs)), snrs)
                ax.set_xlabel('Frequency Component')
                ax.set_ylabel('Signal-to-Noise Ratio')
                ax.set_title('Significant Spectral Peaks')
                ax.set_xticks(range(len(freqs)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='Threshold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Color bars above threshold
                for bar, snr in zip(bars, snrs):
                    if snr > 3.0:
                        bar.set_color('red')
            else:
                ax.text(0.5, 0.5, 'No significant\nspectral peaks\nfound',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Spectral Analysis')
        
        # 6. Foam candidates
        ax = axes[2, 1]
        if 'foam_candidates' in analysis_results and analysis_results['foam_candidates']:
            candidates = analysis_results['foam_candidates'][-10:]  # Last 10 candidates
            if candidates:
                times = [c['time'] for c in candidates]
                z_scores = [c['z_score'] for c in candidates]
                
                bars = ax.bar(range(len(z_scores)), z_scores)
                ax.set_xlabel('Event')
                ax.set_ylabel('Anomaly Z-score')
                ax.set_title('Recent Foam Candidates')
                ax.set_xticks(range(len(z_scores)))
                ax.set_xticklabels([f"Event {i+1}" for i in range(len(z_scores))], rotation=45)
                ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Warning')
                ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Alert')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Color bars by severity
                for bar, z in zip(bars, z_scores):
                    if z > 3.0:
                        bar.set_color('red')
                    elif z > 2.0:
                        bar.set_color('orange')
            else:
                ax.text(0.5, 0.5, 'No foam candidates\nin recent data',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Foam Candidates')
        
        plt.tight_layout()
        
        # Save figure
        report_path = output_path / "analysis_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_report = self.generate_html_report(analysis_results, report_path)
        html_path = output_path / "analysis_report.html"
        with open(html_path, 'w') as f:
            f.write(html_report)
        
        return html_path
    
    def generate_html_report(self, analysis: dict, image_path: Path) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lightning Foam Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .section {{ background: white; padding: 20px; border-radius: 10px; 
                          box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                             gap: 15px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; 
                            border-left: 4px solid #667eea; }}
                .alert {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; 
                        border-radius: 8px; margin: 10px 0; }}
                .critical {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; }}
                .foam-event {{ background: #d1ecf1; border: 1px solid #bee5eb; 
                             padding: 10px; margin: 5px 0; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚡ Lightning Foam Analysis Report</h1>
                <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Strikes</h3>
                        <p style="font-size: 24px; font-weight: bold; color: #667eea;">
                            {analysis.get('total_strikes', 0):,}
                        </p>
                    </div>
                    <div class="stat-card">
                        <h3>Time Range</h3>
                        <p>{analysis.get('time_range', {}).get('duration_hours', 0):.1f} hours</p>
                    </div>
                    <div class="stat-card">
                        <h3>Mean Power</h3>
                        <p>{analysis.get('power_statistics', {}).get('mean', 0):.1f} kA</p>
                    </div>
                    <div class="stat-card">
                        <h3>Max Power</h3>
                        <p>{analysis.get('power_statistics', {}).get('max', 0):.1f} kA</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Analysis Visualization</h2>
                <img src="{image_path.name}" alt="Analysis Charts">
            </div>
        """
        
                # Add spectral analysis section
        if analysis.get('spectral_analysis'):
            html += """
            <div class="section">
                <h2>📡 Spectral Analysis</h2>
                <table>
                    <tr>
                        <th>Frequency</th>
                        <th>Value (Hz)</th>
                        <th>SNR</th>
                        <th>Significant</th>
                        <th>Notes</th>
                    </tr>
            """
            
            foam_freqs = {
                '2_mHz': '2.005 mHz (1 AU light-crossing)',
                'lunar_tidal': '0.022 mHz (M2 lunar tide)',
                'schumann_7.8': '7.83 Hz (Schumann fundamental)',
                'schumann_14.3': '14.3 Hz (Schumann harmonic)',
                'schumann_20.8': '20.8 Hz (Schumann harmonic)'
            }
            
            for name, data in analysis['spectral_analysis'].items():
                freq_name = foam_freqs.get(name, name)
                is_significant = data.get('significant', False)
                snr = data.get('snr', 0)
                
                html += f"""
                    <tr>
                        <td>{freq_name}</td>
                        <td>{data.get('frequency_hz', 0):.6f}</td>
                        <td>{snr:.2f}</td>
                        <td>
                """
                
                if is_significant:
                    html += '<span style="color: red; font-weight: bold;">YES ⚠️</span>'
                else:
                    html += '<span style="color: green;">No</span>'
                
                html += f"""
                        </td>
                        <td>
                """
                
                if is_significant:
                    if 'schumann' in name:
                        html += 'Possible Schumann coupling detected'
                    elif 'mHz' in name:
                        html += 'Possible spacetime foam resonance'
                    else:
                        html += 'Significant peak detected'
                else:
                    html += 'Below detection threshold'
                
                html += """
                        </td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        # Add foam candidates section
        if analysis.get('foam_candidates'):
            candidates = analysis['foam_candidates'][-5:]  # Last 5 candidates
            
            html += """
            <div class="section">
                <h2>🌀 Recent Foam Event Candidates</h2>
            """
            
            for i, candidate in enumerate(reversed(candidates), 1):
                z_score = candidate.get('z_score', 0)
                alert_class = "critical" if z_score > 3.0 else "warning" if z_score > 2.0 else ""
                
                html += f"""
                <div class="foam-event {alert_class}">
                    <h3>Candidate #{i} - {candidate.get('time', 'Unknown')}</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h4>Strikes</h4>
                            <p>{candidate.get('n_strikes', 0)}</p>
                        </div>
                        <div class="stat-card">
                            <h4>Avg Power</h4>
                            <p>{candidate.get('avg_power', 0):.1f} kA</p>
                        </div>
                        <div class="stat-card">
                            <h4>Z-Score</h4>
                            <p>{z_score:.2f}</p>
                        </div>
                        <div class="stat-card">
                            <h4>Location</h4>
                            <p>{candidate.get('center_lat', 0):.2f}°, {candidate.get('center_lon', 0):.2f}°</p>
                        </div>
                    </div>
                """
                
                if z_score > 3.0:
                    html += """
                    <div class="alert critical">
                        <strong>CRITICAL:</strong> High probability of spacetime foam disturbance detected!
                    </div>
                    """
                elif z_score > 2.0:
                    html += """
                    <div class="alert warning">
                        <strong>WARNING:</strong> Possible foam disturbance detected.
                    </div>
                    """
                
                html += """
                </div>
                """
            
            html += """
            </div>
            """
        
        # Add temporal patterns section
        if analysis.get('temporal_patterns'):
            tp = analysis['temporal_patterns']
            
            html += """
            <div class="section">
                <h2>⏱️ Temporal Patterns</h2>
                <div class="stats-grid">
            """
            
            html += f"""
                    <div class="stat-card">
                        <h4>Mean Interval</h4>
                        <p>{tp.get('mean_interval_s', 0):.1f} s</p>
                    </div>
                    <div class="stat-card">
                        <h4>Interval Std Dev</h4>
                        <p>{tp.get('std_interval_s', 0):.1f} s</p>
                    </div>
                    <div class="stat-card">
                        <h4>Coefficient of Variation</h4>
                        <p>{tp.get('cv_interval', 0):.3f}</p>
                    </div>
            """
            
            if tp.get('regular_intervals', False):
                html += f"""
                    <div class="stat-card" style="border-left-color: #dc3545;">
                        <h4>Regular Intervals</h4>
                        <p style="color: #dc3545; font-weight: bold;">DETECTED ⚠️</p>
                        <p>Period: {tp.get('estimated_period_s', 0):.1f} s</p>
                    </div>
                """
            else:
                html += """
                    <div class="stat-card" style="border-left-color: #28a745;">
                        <h4>Regular Intervals</h4>
                        <p style="color: #28a745;">Not detected</p>
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        # Add spatial clusters section
        if analysis.get('spatial_clusters'):
            sc = analysis['spatial_clusters']
            
            html += f"""
            <div class="section">
                <h2>📍 Spatial Clusters</h2>
                <p><strong>Number of clusters detected:</strong> {sc.get('n_clusters', 0)}</p>
            """
            
            if sc.get('cluster_sizes'):
                html += """
                <table>
                    <tr>
                        <th>Cluster</th>
                        <th>Size</th>
                        <th>Center Latitude</th>
                        <th>Center Longitude</th>
                    </tr>
                """
                
                for i, (size, center) in enumerate(zip(sc['cluster_sizes'], sc['cluster_centers']), 1):
                    html += f"""
                    <tr>
                        <td>#{i}</td>
                        <td>{size} strikes</td>
                        <td>{center[0]:.4f}°</td>
                        <td>{center[1]:.4f}°</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            html += """
            </div>
            """
        
        # Add recommendations section
        html += """
        <div class="section">
            <h2>🎯 Recommendations</h2>
            <div class="alert">
                <h3>Based on Analysis:</h3>
                <ul>
        """
        
        recommendations = []
        
        # Check for significant findings
        if analysis.get('spectral_analysis'):
            sig_peaks = [k for k, v in analysis['spectral_analysis'].items() 
                        if v.get('significant', False)]
            
            if sig_peaks:
                rec = f"Found {len(sig_peaks)} significant spectral peaks. "
                if any('schumann' in p for p in sig_peaks):
                    rec += "Schumann resonance coupling detected - investigate further."
                elif any('mHz' in p for p in sig_peaks):
                    rec += "Low-frequency peaks detected - possible spacetime foam resonance."
                recommendations.append(rec)
        
        if analysis.get('foam_candidates'):
            critical = [c for c in analysis['foam_candidates'] if c.get('z_score', 0) > 3.0]
            if critical:
                recommendations.append(f"Found {len(critical)} critical foam events. Review detailed logs.")
        
        if analysis.get('temporal_patterns', {}).get('regular_intervals', False):
            period = analysis['temporal_patterns'].get('estimated_period_s', 0)
            recommendations.append(f"Regular temporal intervals detected (period: {period:.1f}s). This suggests resonant behavior.")
        
        if not recommendations:
            recommendations.append("No significant anomalies detected. Continue monitoring.")
        
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
                <h3>Next Steps:</h3>
                <ol>
                    <li>Review raw data for the most recent foam candidates</li>
                    <li>Cross-reference with Schumann resonance data from Tomsk/Cumiana</li>
                    <li>Check for solar flare activity during detected events</li>
                    <li>Look for correlations with lunar phase/tidal cycles</li>
                    <li>Run deeper spectral analysis on clustered events</li>
                </ol>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Data Quality</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Status</th>
                    <th>Notes</th>
                </tr>
        """
        
        # Data quality checks
        quality_checks = [
            ("Total Strikes", analysis.get('total_strikes', 0) > 100, 
             "At least 100 strikes recommended for analysis"),
            ("Time Coverage", analysis.get('time_range', {}).get('duration_hours', 0) > 1,
             "At least 1 hour of data recommended"),
            ("Spatial Coverage", analysis.get('spatial_distribution', {}).get('lat_range', [0, 0])[1] - 
             analysis.get('spatial_distribution', {}).get('lat_range', [0, 0])[0] > 10,
             "Wide spatial distribution improves analysis"),
            ("Power Range", analysis.get('power_statistics', {}).get('std', 0) > 5,
             "Good power variability detected"),
        ]
        
        for check_name, passed, note in quality_checks:
            status = "✅ PASS" if passed else "⚠️ WARNING"
            color = "#28a745" if passed else "#ffc107"
            
            html += f"""
                <tr>
                    <td>{check_name}</td>
                    <td style="color: {color}; font-weight: bold;">{status}</td>
                    <td>{note}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        
        <div class="section" style="text-align: center; color: #666; font-size: 0.9em;">
            <p>Generated by Lightning Foam Detector v1.0.0</p>
            <p>Output directory: """ + str(self.data_dir) + """</p>
            <p>Analysis timestamp: """ + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC') + """</p>
        </div>
        
        </body>
        </html>
        """
        
        return html

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

import argparse

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description='Lightning Foam Detector - Monitor global lightning for spacetime foam disturbances'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Config.OUTPUT_DIR),
        help='Output directory for data and reports'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis on collected data instead of streaming'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to analyze (default: 7)'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate HTML report after analysis'
    )
    
    parser.add_argument(
        '--websocket-only',
        action='store_true',
        help='Use only WebSocket (no MQTT)'
    )
    
    parser.add_argument(
        '--mqtt-only',
        action='store_true',
        help='Use only MQTT (no WebSocket)'
    )
    
    args = parser.parse_args()
    
    # Update config if custom output directory specified
    if args.output_dir != str(Config.OUTPUT_DIR):
        Config.OUTPUT_DIR = Path(args.output_dir)
    
    if args.analyze:
        # Run analysis mode
        analyzer = FoamAnalyzer(Config.OUTPUT_DIR)
        
        print(f"Loading data from {Config.OUTPUT_DIR}...")
        df = analyzer.load_strikes(args.days)
        
        if df.empty:
            print("No data found for analysis.")
            return
        
        print(f"Analyzing {len(df)} strikes from the last {args.days} days...")
        analysis = analyzer.analyze_foam_patterns(df)
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"Total strikes: {analysis['total_strikes']:,}")
        print(f"Time range: {analysis['time_range']['duration_hours']:.1f} hours")
        print(f"Mean power: {analysis['power_statistics']['mean']:.1f} kA")
        
        if analysis.get('foam_candidates'):
            print(f"\nFoam candidates found: {len(analysis['foam_candidates'])}")
            for candidate in analysis['foam_candidates'][-3:]:
                print(f"  - {candidate['time']}: {candidate['n_strikes']} strikes, "
                      f"Z-score: {candidate['z_score']:.2f}")
        
        if analysis.get('spectral_analysis'):
            print("\nSpectral analysis:")
            for name, data in analysis['spectral_analysis'].items():
                if data.get('significant', False):
                    print(f"  - {name}: SNR = {data['snr']:.2f} (SIGNIFICANT)")
        
        if args.generate_report:
            print("\nGenerating HTML report...")
            report_path = analyzer.generate_report(analysis, Config.OUTPUT_DIR)
            print(f"Report saved to: {report_path}")
            
            # Open in browser (optional)
            import webbrowser
            webbrowser.open(f"file://{report_path}")
        
    else:
        # Run streaming mode
        print("="*60)
        print("LIGHTNING FOAM DETECTOR")
        print("="*60)
        print(f"Output directory: {Config.OUTPUT_DIR}")
        print(f"WebSocket servers: {', '.join(Config.WEBSOCKET_SERVERS)}")
        print(f"MQTT broker: {Config.MQTT_BROKER}")
        print(f"Foam frequencies monitored: {list(Config.FOAM_FREQUENCIES.keys())}")
        print("="*60)
        print("Starting data collection...")
        print("Press Ctrl+C to stop\n")
        
        detector = LightningFoamDetector()
        
        try:
            # Run the main application
            asyncio.run(detector.run())
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
            detector.stop()
        except Exception as e:
            print(f"Fatal error: {e}")
            detector.stop()
            raise

# ============================================================================
# INSTALLATION AND SETUP SCRIPT
# ============================================================================

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """aiohttp>=3.8.0
websockets>=11.0.0
paho-mqtt>=1.6.1
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("Created requirements.txt")

def create_service_file():
    """Create systemd service file for Linux"""
    service = """[Unit]
Description=Lightning Foam Detector
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=%i
ExecStart=/usr/bin/python3 /path/to/lightning_foam_detector.py
WorkingDirectory=/path/to/
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    with open("lightning-foam-detector.service", "w") as f:
        f.write(service)
    
    print("Created lightning-foam-detector.service")
    print("To install: sudo cp lightning-foam-detector.service /etc/systemd/system/")
    print("To enable: sudo systemctl enable lightning-foam-detector")
    print("To start: sudo systemctl start lightning-foam-detector")

def create_windows_service():
    """Create Windows service script"""
    script = """import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import subprocess
import os

class LightningFoamService(win32serviceutil.ServiceFramework):
    _svc_name_ = "LightningFoamDetector"
    _svc_display_name_ = "Lightning Foam Detector"
    _svc_description_ = "Monitors global lightning for spacetime foam disturbances"

        def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_running = False

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                            servicemanager.PYS_SERVICE_STARTED,
                            (self._svc_name_, ''))
        self.main()

    def main(self):
        # Run the detector
        detector = LightningFoamDetector()
        import asyncio
        asyncio.run(detector.run())

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(LightningFoamService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(LightningFoamService)
"""
    
    with open("windows_service.py", "w") as f:
        f.write(script)
    
    print("Created windows_service.py")
    print("To install: python windows_service.py install")
    print("To start: python windows_service.py start")

# ============================================================================
# DATA EXPORT AND INTEGRATION FUNCTIONS
# ============================================================================

class DataExporter:
    """Export data for integration with other systems"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def export_to_csv(self, output_path: Path, days: int = 1):
        """Export recent strikes to CSV"""
        analyzer = FoamAnalyzer(self.data_dir)
        df = analyzer.load_strikes(days)
        
        if df.empty:
            print("No data to export")
            return None
        
        # Select and rename columns
        export_df = df[['timestamp', 'latitude', 'longitude', 'power', 'polarity', 'region', 'source']].copy()
        export_df['datetime'] = pd.to_datetime(export_df['timestamp'], unit='s')
        export_df = export_df[['datetime', 'timestamp', 'latitude', 'longitude', 
                              'power', 'polarity', 'region', 'source']]
        
        export_df.to_csv(output_path, index=False)
        print(f"Exported {len(export_df)} strikes to {output_path}")
        return output_path
    
    def export_foam_events(self, output_path: Path):
        """Export foam events to JSON"""
        events = []
        pattern = self.data_dir / "foam_events" / "event_*.json"
        
        for filepath in pattern.parent.glob(pattern.name):
            try:
                with open(filepath, 'r') as f:
                    event_data = json.load(f)
                    events.append(event_data)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        if events:
            with open(output_path, 'w') as f:
                json.dump(events, f, indent=2)
            print(f"Exported {len(events)} foam events to {output_path}")
            return output_path
        else:
            print("No foam events to export")
            return None
    
    def export_for_schumann_correlation(self, output_path: Path, days: int = 1):
        """Export data formatted for correlation with Schumann resonance data"""
        analyzer = FoamAnalyzer(self.data_dir)
        df = analyzer.load_strikes(days)
        
        if df.empty:
            return None
        
        # Format for Schumann correlation analysis
        # Group by minute and calculate statistics
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['minute'] = df['datetime'].dt.floor('min')
        
        grouped = df.groupby('minute').agg({
            'power': ['count', 'mean', 'std', 'max'],
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['timestamp', 'strike_count', 'power_mean', 'power_std', 
                         'power_max', 'lat_mean', 'lon_mean']
        
        # Convert timestamp to UNIX for easier correlation
        grouped['unix_time'] = grouped['timestamp'].astype('int64') // 10**9
        
        # Save to CSV
        grouped[['unix_time', 'strike_count', 'power_mean', 'power_std', 
                'power_max', 'lat_mean', 'lon_mean']].to_csv(output_path, index=False)
        
        print(f"Exported {len(grouped)} minutes of data for Schumann correlation to {output_path}")
        return output_path

# ============================================================================
# REAL-TIME ALERT SYSTEM
# ============================================================================

class AlertSystem:
    """Real-time alert system for foam events"""
    
    def __init__(self, config: Config):
        self.config = config
        self.alert_history = deque(maxlen=1000)
        self.logger = logging.getLogger("AlertSystem")
        
        # Alert thresholds
        self.thresholds = {
            'power_zscore': 3.0,      # 3 sigma
            'cluster_size': 10,        # strikes in 1 minute
            'temporal_regularity': 0.3, # CV threshold
            'spectral_snr': 3.0        # Signal-to-noise ratio
        }
        
        # Alert methods (can be extended)
        self.alert_methods = []
    
    def register_alert_method(self, method):
        """Register an alert method (email, webhook, etc.)"""
        self.alert_methods.append(method)
    
    def check_foam_event(self, event: FoamEvent):
        """Check if a foam event should trigger alerts"""
        alerts = []
        
        # Check anomaly score
        if event.anomaly_score > 5.0:
            alerts.append({
                'type': 'CRITICAL',
                'message': f'High anomaly foam event detected: {event.event_id}',
                'score': event.anomaly_score,
                'event': event.to_dict()
            })
        
        # Check frequency components
        for freq_name, power in event.frequency_components.items():
            if 'schumann' in freq_name and power > self.thresholds['spectral_snr']:
                alerts.append({
                    'type': 'SCHUMANN',
                    'message': f'Schumann resonance coupling detected: {freq_name}',
                    'frequency': freq_name,
                    'snr': power,
                    'event': event.to_dict()
                })
            elif 'mHz' in freq_name and power > self.thresholds['spectral_snr']:
                alerts.append({
                    'type': 'LOW_FREQ',
                    'message': f'Low-frequency resonance detected: {freq_name}',
                    'frequency': freq_name,
                    'snr': power,
                    'event': event.to_dict()
                })
        
        # Check energy level
        if event.total_energy > 1000:  # Arbitrary threshold
            alerts.append({
                'type': 'HIGH_ENERGY',
                'message': f'High energy foam event: {event.total_energy:.0f} kA·s',
                'energy': event.total_energy,
                'event': event.to_dict()
            })
        
        # Trigger alerts
        for alert in alerts:
            self.trigger_alert(alert)
        
        return alerts
    
    def trigger_alert(self, alert: dict):
        """Trigger all registered alert methods"""
        self.alert_history.append({
            'timestamp': time.time(),
            'alert': alert
        })
        
        self.logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
        
        for method in self.alert_methods:
            try:
                method(alert)
            except Exception as e:
                self.logger.error(f"Alert method failed: {e}")
    
    def save_alert_history(self):
        """Save alert history to file"""
        history_path = self.config.OUTPUT_DIR / "alerts" / "alert_history.jsonl"
        history_path.parent.mkdir(exist_ok=True)
        
        with open(history_path, 'a') as f:
            for entry in self.alert_history:
                f.write(json.dumps(entry) + '\n')

# ============================================================================
# WEB DASHBOARD (OPTIONAL)
# ============================================================================

class WebDashboard:
    """Simple web dashboard for real-time monitoring"""
    
    def __init__(self, detector: LightningFoamDetector, port: int = 8080):
        self.detector = detector
        self.port = port
        self.app = None
        self.logger = logging.getLogger("WebDashboard")
    
    async def start(self):
        """Start the web dashboard"""
        from aiohttp import web
        import aiohttp_jinja2
        import jinja2
        
        self.app = web.Application()
        
        # Setup Jinja2 templates
        aiohttp_jinja2.setup(
            self.app, 
            loader=jinja2.FileSystemLoader(str(Path(__file__).parent / 'templates'))
        )
        
        # Add routes
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/api/stats', self.handle_stats)
        self.app.router.add_get('/api/recent', self.handle_recent)
        self.app.router.add_get('/api/events', self.handle_events)
        self.app.router.add_get('/ws', self.handle_websocket)
        
        # Static files
        self.app.router.add_static('/static', str(Path(__file__).parent / 'static'))
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        self.logger.info(f"Web dashboard started on http://localhost:{self.port}")
    
    async def handle_index(self, request):
        """Serve main dashboard page"""
        return aiohttp_jinja2.render_template('index.html', request, {})
    
    async def handle_stats(self, request):
        """API endpoint for statistics"""
        stats = {
            'total_strikes': self.detector.storage.get_strike_count(),
            'foam_events': self.detector.detector.event_counter,
            'uptime': time.time() - self.detector.start_time if hasattr(self.detector, 'start_time') else 0,
            'last_update': datetime.utcnow().isoformat()
        }
        return web.json_response(stats)
    
    async def handle_recent(self, request):
        """API endpoint for recent strikes"""
        analyzer = FoamAnalyzer(self.detector.config.OUTPUT_DIR)
        df = analyzer.load_strikes(1)  # Last 24 hours
        
        if df.empty:
            return web.json_response({'strikes': []})
        
        # Get last 100 strikes
        recent = df.tail(100).to_dict('records')
        return web.json_response({'strikes': recent})
    
    async def handle_events(self, request):
        """API endpoint for foam events"""
        events = []
        pattern = self.detector.config.OUTPUT_DIR / "foam_events" / "event_*.json"
        
        for filepath in sorted(pattern.parent.glob(pattern.name), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            try:
                with open(filepath, 'r') as f:
                    events.append(json.load(f))
            except:
                pass
        
        return web.json_response({'events': events})
    
    async def handle_websocket(self, request):
        """WebSocket endpoint for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.logger.info("WebSocket connection established")
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle incoming messages
                    data = json.loads(msg.data)
                    # Process command if needed
                    
                    # Send periodic updates
                    while not ws.closed:
                        # Get latest stats
                        stats = {
                            'timestamp': time.time(),
                            'strikes_last_minute': self.get_recent_strike_count(60),
                            'foam_events_today': self.get_today_event_count(),
                            'alerts': self.get_recent_alerts()
                        }
                        
                        await ws.send_json(stats)
                        await asyncio.sleep(5)  # Update every 5 seconds
                        
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.logger.info("WebSocket connection closed")
        
        return ws
    
    def get_recent_strike_count(self, seconds: int):
        """Get strike count in last N seconds"""
        # Implementation depends on how you track real-time data
        return 0
    
    def get_today_event_count(self):
        """Get foam event count today"""
        # Implementation depends on your data storage
        return 0
    
    def get_recent_alerts(self):
        """Get recent alerts"""
        # Implementation depends on your alert system
        return []

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create necessary directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("Setting up Lightning Foam Detector...")
        create_requirements_file()
        create_service_file()
        print("\nSetup complete!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: python lightning_foam_detector.py")
        print("3. Or run in background: python lightning_foam_detector.py --daemon")
        sys.exit(0)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        exporter = DataExporter(Config.OUTPUT_DIR)
        
        if len(sys.argv) > 2 and sys.argv[2] == "csv":
            output_path = Config.OUTPUT_DIR / "export" / f"strikes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path.parent.mkdir(exist_ok=True)
            exporter.export_to_csv(output_path, days=7)
        
        elif len(sys.argv) > 2 and sys.argv[2] == "schumann":
            output_path = Config.OUTPUT_DIR / "export" / f"schumann_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path.parent.mkdir(exist_ok=True)
            exporter.export_for_schumann_correlation(output_path, days=1)
        
        elif len(sys.argv) > 2 and sys.argv[2] == "events":
            output_path = Config.OUTPUT_DIR / "export" / f"foam_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path.parent.mkdir(exist_ok=True)
            exporter.export_foam_events(output_path)
        
        else:
            print("Usage: python lightning_foam_detector.py export [csv|schumann|events]")
        
        sys.exit(0)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # Run analysis mode
        analyzer = FoamAnalyzer(Config.OUTPUT_DIR)
        
        days = 7
        if len(sys.argv) > 2:
            try:
                days = int(sys.argv[2])
            except ValueError:
                pass
        
        print(f"Analyzing data from last {days} days...")
        df = analyzer.load_strikes(days)
        
        if df.empty:
            print("No data found.")
            sys.exit(1)
        
        analysis = analyzer.analyze_foam_patterns(df)
        
        # Generate report
        report_path = analyzer.generate_report(analysis, Config.OUTPUT_DIR)
        print(f"\nReport generated: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total strikes analyzed: {analysis['total_strikes']:,}")
        print(f"Time period: {analysis['time_range']['duration_hours']:.1f} hours")
        print(f"Mean strike power: {analysis['power_statistics']['mean']:.1f} kA")
        
        # Check for significant findings
        significant_events = []
        
        if analysis.get('foam_candidates'):
            high_z = [c for c in analysis['foam_candidates'] if c.get('z_score', 0) > 3.0]
            if high_z:
                significant_events.append(f"{len(high_z)} high-Z foam candidates (Z > 3.0)")
        
        if analysis.get('spectral_analysis'):
            sig_peaks = [k for k, v in analysis['spectral_analysis'].items() 
                        if v.get('significant', False)]
            if sig_peaks:
                significant_events.append(f"{len(sig_peaks)} significant spectral peaks")
        
        if analysis.get('temporal_patterns', {}).get('regular_intervals', False):
            significant_events.append("Regular temporal intervals detected")
        
        if significant_events:
            print("\nSIGNIFICANT FINDINGS:")
            for event in significant_events:
                print(f"  • {event}")
        else:
            print("\nNo significant anomalies detected.")
        
        print("\n" + "="*60)
        
        sys.exit(0)
    
    else:
        # Run main application
        main()
