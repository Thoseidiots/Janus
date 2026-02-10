use wasmtime::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WasmSnapshot {
    pub counter: i32,
    pub step: i32,
}

pub struct JanusWasmHost {
    pub engine: Engine,
    pub module: Module,
    pub store: Store<WasmSnapshot>,
    pub instance: Instance,
}

impl JanusWasmHost {
    pub fn new(wasm_binary: &[u8]) -> Result<Self> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_binary)?;
        let mut store = Store::new(&engine, WasmSnapshot { counter: 0, step: 0 });
        
        let instance = Instance::new(&mut store, &module, &[])?;
        
        Ok(Self {
            engine,
            module,
            store,
            instance,
        })
    }

    pub fn tick(&mut self) -> Result<i32> {
        let tick_fn = self.instance.get_typed_func::<(), i32>(&mut self.store, "tick")?;
        let result = tick_fn.call(&mut self.store, ())?;
        Ok(result)
    }

    pub fn get_snapshot(&self) -> WasmSnapshot {
        self.store.data().clone()
    }

    pub fn restore_snapshot(&mut self, snapshot: WasmSnapshot) {
        *self.store.data_mut() = snapshot;
    }
}
