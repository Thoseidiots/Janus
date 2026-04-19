"""
Janus Auto Launcher
Automatically starts and manages all Janus systems
Runs everything smoothly in the background
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime

class JanusAutoLauncher:
    """
    Automatically launches and manages all Janus systems
    """
    
    def __init__(self):
        self.workspace_root = Path.cwd()
        self.systems = {}
        self.running = False
        self.startup_log = []
        
        print("🤖 Janus Auto Launcher Starting...")
        print(f"📁 Workspace: {self.workspace_root}")
    
    def log(self, message: str):
        """Log startup messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.startup_log.append(log_entry)
    
    def check_dependencies(self) -> bool:
        """Check if required files exist"""
        self.log("🔍 Checking dependencies...")
        
        required_files = [
            "janus_human_capable.py",
            "advanced_3d_face_generator.py", 
            "janus_integration_hub.py",
            "hardware_sense.py",
            "window_manager.py",
            "vision_action_pipeline.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.workspace_root / file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.log(f"❌ Missing files: {missing_files}")
            return False
        
        self.log("✅ All required files found")
        return True
    
    def start_hardware_monitoring(self):
        """Start hardware monitoring in background"""
        try:
            self.log("🖥️ Starting hardware monitoring...")
            
            # Import and start hardware systems
            from hardware_sense import HardwareAwareness
            from hardware_events import HardwareEventDetector
            
            self.systems['hardware_awareness'] = HardwareAwareness()
            self.systems['hardware_events'] = HardwareEventDetector(
                self.systems['hardware_awareness'].sense
            )
            
            # Start monitoring
            self.systems['hardware_events'].start_monitoring()
            
            self.log("✅ Hardware monitoring active")
            return True
            
        except Exception as e:
            self.log(f"⚠️ Hardware monitoring failed: {e}")
            return False
    
    def start_face_generation_system(self):
        """Start 3D face generation system"""
        try:
            self.log("🎭 Starting 3D face generation...")
            
            from enhanced_3d_face_generator import AAA_FaceGenerator
            self.systems['face_generator'] = AAA_FaceGenerator()
            
            # Test generation
            test_char = self.systems['face_generator'].generate_character_face(
                archetype='human',
                performance_target='console'
            )
            
            self.log(f"✅ Face generator ready - test character: {test_char.performance_stats['polygon_count']} triangles")
            return True
            
        except Exception as e:
            self.log(f"⚠️ Face generation failed: {e}")
            # Try fallback to basic generator
            try:
                from advanced_3d_face_generator import ProceduralFaceGenerator
                self.systems['face_generator'] = ProceduralFaceGenerator()
                self.log("✅ Basic face generator ready")
                return True
            except Exception as e2:
                self.log(f"❌ All face generators failed: {e2}")
                return False
    
    def start_integration_hub(self):
        """Start the integration hub"""
        try:
            self.log("🔗 Starting integration hub...")
            
            from janus_integration_hub import JanusIntegrationHub
            self.systems['integration_hub'] = JanusIntegrationHub()
            
            self.log("✅ Integration hub ready")
            return True
            
        except Exception as e:
            self.log(f"⚠️ Integration hub failed: {e}")
            return False
    
    def start_training_systems(self):
        """Start training data generation"""
        try:
            self.log("📊 Starting training systems...")
            
            from janus_training_connector import JanusTrainingConnector
            self.systems['training_connector'] = JanusTrainingConnector()
            
            self.log("✅ Training systems ready")
            return True
            
        except Exception as e:
            self.log(f"⚠️ Training systems failed: {e}")
            return False
    
    def start_janus_core(self):
        """Start core Janus system"""
        try:
            self.log("🤖 Starting Janus core system...")
            
            from janus_human_capable import JanusHumanCapable
            self.systems['janus_core'] = JanusHumanCapable()
            
            # Verify capabilities
            capabilities = self.systems['janus_core'].verify_capabilities()
            active_caps = sum(1 for v in capabilities.values() if v)
            
            self.log(f"✅ Janus core ready - {active_caps}/{len(capabilities)} capabilities active")
            return True
            
        except Exception as e:
            self.log(f"⚠️ Janus core failed: {e}")
            return False
    
    def connect_to_game_database(self):
        """Connect to game AI database if available"""
        try:
            game_db_path = self.workspace_root / "game_ai_database"
            if game_db_path.exists():
                self.log("🎮 Connecting to game AI database...")
                
                # Add to Python path
                sys.path.insert(0, str(game_db_path))
                
                # Try to import and connect
                from generators.character_generator import CharacterGenerator
                self.systems['game_db_character_gen'] = CharacterGenerator()
                
                self.log("✅ Game AI database connected")
                return True
            else:
                self.log("ℹ️ Game AI database not found - skipping")
                return True
                
        except Exception as e:
            self.log(f"⚠️ Game database connection failed: {e}")
            return True  # Non-critical
    
    def start_background_services(self):
        """Start background services"""
        self.log("🔄 Starting background services...")
        
        def background_loop():
            while self.running:
                try:
                    # Generate training data periodically
                    if 'training_connector' in self.systems:
                        if hasattr(self.systems['training_connector'], 'hub') and self.systems['training_connector'].hub:
                            # Generate a small batch every 5 minutes
                            self.systems['training_connector'].generate_character_training_batch(count=2)
                    
                    # Monitor hardware and adapt
                    if 'hardware_events' in self.systems:
                        events = self.systems['hardware_events'].get_events(timeout=1.0)
                        if events:
                            self.log(f"🔔 {len(events)} hardware events detected")
                    
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    self.log(f"⚠️ Background service error: {e}")
                    time.sleep(60)
        
        self.background_thread = threading.Thread(target=background_loop, daemon=True)
        self.background_thread.start()
        
        self.log("✅ Background services started")
    
    def launch_all_systems(self):
        """Launch all Janus systems automatically"""
        self.log("🚀 JANUS AUTO LAUNCHER STARTING")
        self.log("=" * 50)
        
        self.running = True
        startup_success = True
        
        # Check dependencies first
        if not self.check_dependencies():
            self.log("❌ Dependency check failed - cannot start")
            return False
        
        # Start systems in order
        startup_sequence = [
            ("Hardware Monitoring", self.start_hardware_monitoring),
            ("Face Generation", self.start_face_generation_system),
            ("Janus Core", self.start_janus_core),
            ("Integration Hub", self.start_integration_hub),
            ("Training Systems", self.start_training_systems),
            ("Game Database", self.connect_to_game_database)
        ]
        
        for system_name, start_func in startup_sequence:
            self.log(f"🔄 Starting {system_name}...")
            success = start_func()
            if not success:
                self.log(f"⚠️ {system_name} failed to start - continuing anyway")
                startup_success = False
            time.sleep(1)  # Brief pause between systems
        
        # Start background services
        self.start_background_services()
        
        # Show final status
        self.log("=" * 50)
        if startup_success:
            self.log("🎉 ALL SYSTEMS OPERATIONAL")
        else:
            self.log("⚠️ SOME SYSTEMS FAILED - PARTIAL OPERATION")
        
        self.log(f"🔧 Active systems: {len(self.systems)}")
        for system_name in self.systems.keys():
            self.log(f"  ✅ {system_name}")
        
        self.log("🤖 Janus is now running automatically!")
        self.log("=" * 50)
        
        return True
    
    def run_demo_sequence(self):
        """Run an automated demo of all systems"""
        if not self.systems:
            self.log("❌ No systems running - cannot run demo")
            return
        
        self.log("🎬 RUNNING AUTOMATED DEMO")
        self.log("-" * 30)
        
        # Demo 1: Generate characters automatically
        if 'integration_hub' in self.systems:
            self.log("🎭 Demo: Generating characters...")
            
            demo_characters = [
                "heroic knight",
                "wise wizard", 
                "elven archer",
                "battle-scarred orc"
            ]
            
            for char_desc in demo_characters:
                try:
                    result = self.systems['integration_hub'].generate_character_with_context(char_desc)
                    if result['success']:
                        vertex_count = result['character_data']['metadata']['vertex_count']
                        self.log(f"  ✅ {char_desc}: {vertex_count} vertices")
                    else:
                        self.log(f"  ❌ {char_desc}: failed")
                except Exception as e:
                    self.log(f"  ❌ {char_desc}: error - {e}")
                
                time.sleep(2)
        
        # Demo 2: Generate training data
        if 'training_connector' in self.systems:
            self.log("📊 Demo: Generating training data...")
            try:
                batch = self.systems['training_connector'].generate_character_training_batch(count=3)
                self.log(f"  ✅ Generated {len(batch)} training samples")
            except Exception as e:
                self.log(f"  ❌ Training generation failed: {e}")
        
        # Demo 3: Show system status
        self.log("📋 Demo: System status...")
        if 'janus_core' in self.systems:
            try:
                status = self.systems['janus_core'].get_system_status()
                self.log(f"  🖥️ Monitors: {len(status.get('monitors', []))}")
                self.log(f"  🪟 Windows: {len(status.get('windows', []))}")
                if 'hardware_feeling' in status:
                    self.log(f"  🤖 Hardware feeling: {status['hardware_feeling'][:50]}...")
            except Exception as e:
                self.log(f"  ❌ Status check failed: {e}")
        
        self.log("🎬 Demo complete!")
    
    def keep_running(self):
        """Keep the launcher running and responsive"""
        self.log("🔄 Janus running in background...")
        self.log("💡 Press Ctrl+C to stop")
        
        try:
            while self.running:
                time.sleep(10)
                
                # Periodic status check
                if len(self.systems) > 0:
                    self.log(f"💓 Heartbeat - {len(self.systems)} systems active")
                
        except KeyboardInterrupt:
            self.log("🛑 Shutdown requested...")
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all systems"""
        self.log("🛑 SHUTTING DOWN JANUS SYSTEMS")
        
        self.running = False
        
        # Stop hardware monitoring
        if 'hardware_events' in self.systems:
            try:
                self.systems['hardware_events'].stop_monitoring()
                self.log("✅ Hardware monitoring stopped")
            except Exception:
                pass
        
        # Stop other systems
        for system_name in list(self.systems.keys()):
            try:
                if hasattr(self.systems[system_name], 'shutdown'):
                    self.systems[system_name].shutdown()
                del self.systems[system_name]
                self.log(f"✅ {system_name} stopped")
            except Exception:
                pass
        
        self.log("🤖 Janus shutdown complete")


def main():
    """Main launcher function"""
    launcher = JanusAutoLauncher()
    
    # Launch all systems
    success = launcher.launch_all_systems()
    
    if success:
        # Run demo
        launcher.run_demo_sequence()
        
        # Keep running
        launcher.keep_running()
    else:
        print("❌ Failed to start Janus systems")


if __name__ == "__main__":
    main()