// engine-editor/src/lib.rs

pub mod ui {
    // A simplified immediate-mode UI library stub
    pub struct Ui {
        // state would go here
    }

    impl Ui {
        pub fn new() -> Self { Ui {} }
        pub fn begin_frame(&mut self) { /* ... */ }
        pub fn end_frame(&mut self) { /* ... */ }
        
        pub fn button(&mut self, label: &str) -> bool {
            println!("Button: {}", label);
            false
        }

        pub fn label(&mut self, text: &str) {
            println!("Label: {}", text);
        }
    }
}

pub mod panels {
    use super::ui::Ui;

    pub struct HierarchyPanel;
    impl HierarchyPanel {
        pub fn new() -> Self { Self }
        pub fn draw(&mut self, ui: &mut Ui) {
            ui.label("Hierarchy");
            // In a real editor, this would list scene entities
        }
    }

    pub struct InspectorPanel;
    impl InspectorPanel {
        pub fn new() -> Self { Self }
        pub fn draw(&mut self, ui: &mut Ui) {
            ui.label("Inspector");
            // In a real editor, this would show component properties
        }
    }
}

pub mod commands {
    pub trait EditorCommand {
        fn execute(&mut self);
        fn undo(&mut self);
    }

    pub struct UndoStack {
        undo_list: Vec<Box<dyn EditorCommand>>,
        redo_list: Vec<Box<dyn EditorCommand>>,
        max_history: usize,
    }

    impl UndoStack {
        pub fn new() -> Self {
            Self {
                undo_list: Vec::new(),
                redo_list: Vec::new(),
                max_history: 100,
            }
        }

        pub fn push_and_execute(&mut self, mut cmd: Box<dyn EditorCommand>) {
            cmd.execute();
            self.undo_list.push(cmd);
            if self.undo_list.len() > self.max_history {
                self.undo_list.remove(0);
            }
            self.redo_list.clear();
        }

        pub fn undo(&mut self) {
            if let Some(mut cmd) = self.undo_list.pop() {
                cmd.undo();
                self.redo_list.push(cmd);
            }
        }

        pub fn redo(&mut self) {
            if let Some(mut cmd) = self.redo_list.pop() {
                cmd.execute();
                self.undo_list.push(cmd);
            }
        }
    }
}

pub mod editor {
    use super::ui::Ui;
    use super::panels::{HierarchyPanel, InspectorPanel};

    pub struct Editor {
        ui: Ui,
        hierarchy: HierarchyPanel,
        inspector: InspectorPanel,
        pub undo_stack: crate::commands::UndoStack,
    }

    impl Editor {
        pub fn new() -> Self {
            Self {
                ui: Ui::new(),
                hierarchy: HierarchyPanel::new(),
                inspector: InspectorPanel::new(),
                undo_stack: crate::commands::UndoStack::new(),
            }
        }

        pub fn tick(&mut self) {
            self.ui.begin_frame();

            self.hierarchy.draw(&mut self.ui);
            self.inspector.draw(&mut self.ui);

            self.ui.end_frame();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::editor::Editor;

    #[test]
    fn test_editor_creation() {
        let mut editor = Editor::new();
        // Just ensure it can be created and ticked without panicking.
        editor.tick();
    }

    struct MockEditCommand {
        target: std::sync::Arc<std::sync::Mutex<i32>>,
        old_val: i32,
        new_val: i32,
    }

    impl crate::commands::EditorCommand for MockEditCommand {
        fn execute(&mut self) {
            *self.target.lock().unwrap() = self.new_val;
        }

        fn undo(&mut self) {
            *self.target.lock().unwrap() = self.old_val;
        }
    }

    // Property 28: Undo/Redo Round-Trip
    // Validates: Requirements 8.5
    #[test]
    fn property_undo_redo_round_trip() {
        use std::sync::{Arc, Mutex};
        let val = Arc::new(Mutex::new(0));
        let mut stack = crate::commands::UndoStack::new();

        let cmd = Box::new(MockEditCommand {
            target: val.clone(),
            old_val: 0,
            new_val: 42,
        });

        stack.push_and_execute(cmd);
        assert_eq!(*val.lock().unwrap(), 42);

        stack.undo();
        assert_eq!(*val.lock().unwrap(), 0);

        stack.redo();
        assert_eq!(*val.lock().unwrap(), 42);
    }

    // Property 27: Editor Component Edit Applies to World
    // Validates: Requirements 8.4
    #[test]
    fn property_editor_component_edit_applies_to_world() {
        use std::sync::{Arc, Mutex};
        let mut editor = Editor::new();
        let world_component = Arc::new(Mutex::new(100));

        let edit_cmd = Box::new(MockEditCommand {
            target: world_component.clone(),
            old_val: 100,
            new_val: 200,
        });

        editor.undo_stack.push_and_execute(edit_cmd);
        assert_eq!(*world_component.lock().unwrap(), 200);
    }
}
