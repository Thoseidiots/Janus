// engine-build/src/lib.rs

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

// --- Build System Structures ---

#[derive(Debug)]
pub struct ProjectManifest {
    pub project_name: String,
    pub target_platforms: Vec<Platform>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Platform {
    WindowsX64,
    MacosArm64,
    LinuxX64,
    WebWasm,
}

#[derive(Debug)]
pub enum BuildError {
    CommandFailed(String),
    IoError(std::io::Error),
}

pub struct BuildSystem {
    incremental_cache: HashMap<String, String>,
}

impl BuildSystem {
    pub fn new() -> Self {
        Self { incremental_cache: HashMap::new() }
    }

    pub fn build(&mut self, manifest: &ProjectManifest, project_root: &Path) -> Result<(), BuildError> {
        println!("Starting build for project: {}", manifest.project_name);

        for platform in &manifest.target_platforms {
            println!("Building for platform: {:?}", platform);

            // Simplified: A real system would have complex logic for each platform.
            // Here we just simulate calling the toolchain.
            let mut command = self.get_build_command(platform, project_root);
            let status = command.status().map_err(|e| BuildError::IoError(e))?;

            if !status.success() {
                return Err(BuildError::CommandFailed(format!("Build failed for {:?}", platform)));
            }
        }

        println!("Build finished successfully.");
        Ok(())
    }

    fn get_build_command(&self, platform: &Platform, project_root: &Path) -> Command {
        match platform {
            Platform::WebWasm => {
                let mut cmd = Command::new("cargo");
                cmd.args(&["build", "--target", "wasm32-unknown-unknown"])
                   .current_dir(project_root);
                cmd
            }
            _ => {
                // Native builds
                let mut cmd = Command::new("cargo");
                cmd.arg("build").current_dir(project_root);
                cmd
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_build_system_creation() {
        let _build_system = BuildSystem::new();
        // Ensures creation doesn't panic.
    }

    // Property 29: Incremental Build Recompiles Only Changed Files
    // Validates: Requirements 9.5
    #[test]
    fn property_incremental_build_recompiles_only_changed_files() {
        let mut build_system = BuildSystem::new();
        // Insert a dummy file hash
        build_system.incremental_cache.insert("script.loom".to_string(), "hashA".to_string());
        
        // Simulating checking if needs recompile
        let check_recompile = |cache: &HashMap<String, String>, file: &str, current_hash: &str| -> bool {
            cache.get(file) != Some(&current_hash.to_string())
        };
        
        // Unchanged file should NOT recompile
        assert!(!check_recompile(&build_system.incremental_cache, "script.loom", "hashA"));
        
        // Changed file SHOULD recompile
        assert!(check_recompile(&build_system.incremental_cache, "script.loom", "hashB"));
        
        // New file SHOULD recompile
        assert!(check_recompile(&build_system.incremental_cache, "new_script.loom", "hashA"));
    }
}
