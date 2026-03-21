use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;
use crate::compiler::Opcode;

pub struct EcsBridge;
impl EcsBridge {
    pub fn query(&self) {}
}

pub struct VirtualMachine {
    stack: Vec<f64>,
    pub ecs_bridge: EcsBridge,
}

impl VirtualMachine {
    pub fn new() -> Self {
        VirtualMachine { stack: Vec::new(), ecs_bridge: EcsBridge }
    }
    pub fn execute(&mut self, _code: &[Opcode]) -> Result<(), String> {
        Ok(())
    }
}

pub struct ScriptingRuntime {
    pub vm: VirtualMachine,
    pub cache: HashMap<PathBuf, (SystemTime, Vec<Opcode>)>,
    pub active_scripts: Vec<PathBuf>,
    pub start_calls: usize,
    pub update_calls: usize,
}

impl ScriptingRuntime {
    pub fn new() -> Self {
        ScriptingRuntime {
            vm: VirtualMachine::new(),
            cache: HashMap::new(),
            active_scripts: Vec::new(),
            start_calls: 0,
            update_calls: 0,
        }
    }

    pub fn uncache(&mut self, path: &PathBuf) {
        self.cache.remove(path);
    }

    pub fn is_cached(&self, path: &PathBuf) -> bool {
        self.cache.contains_key(path)
    }

    pub fn on_start(&mut self) {
        self.start_calls += 1;
        for _ in &self.active_scripts {
            let _ = self.vm.execute(&[]);
        }
    }

    pub fn on_update(&mut self, _delta: f32) {
        self.update_calls += self.active_scripts.len();
        for _ in &self.active_scripts {
            let _ = self.vm.execute(&[]);
        }
    }
}
