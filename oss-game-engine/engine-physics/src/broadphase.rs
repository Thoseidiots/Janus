use crate::types::Aabb;

struct AabbNode {
    aabb: Aabb,
    collider_index: Option<usize>, // Some(_) = leaf
    left: Option<usize>,
    right: Option<usize>,
    parent: Option<usize>,
}

pub struct AabbTree {
    nodes: Vec<AabbNode>,
    root: Option<usize>,
}

impl AabbTree {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), root: None }
    }

    /// Insert a leaf for the given collider with the given AABB.
    pub fn insert(&mut self, collider_index: usize, aabb: Aabb) {
        let leaf = self.nodes.len();
        self.nodes.push(AabbNode {
            aabb,
            collider_index: Some(collider_index),
            left: None,
            right: None,
            parent: None,
        });

        match self.root {
            None => {
                self.root = Some(leaf);
            }
            Some(root) => {
                // Find the best sibling using surface-area heuristic (greedy)
                let sibling = self.find_best_sibling(leaf, root);

                // Create a new internal node to hold sibling + leaf
                let old_parent = self.nodes[sibling].parent;
                let new_internal = self.nodes.len();
                let combined_aabb = self.nodes[sibling].aabb.expand(aabb);
                self.nodes.push(AabbNode {
                    aabb: combined_aabb,
                    collider_index: None,
                    left: Some(sibling),
                    right: Some(leaf),
                    parent: old_parent,
                });

                self.nodes[sibling].parent = Some(new_internal);
                self.nodes[leaf].parent = Some(new_internal);

                match old_parent {
                    None => self.root = Some(new_internal),
                    Some(p) => {
                        if self.nodes[p].left == Some(sibling) {
                            self.nodes[p].left = Some(new_internal);
                        } else {
                            self.nodes[p].right = Some(new_internal);
                        }
                    }
                }

                // Refit ancestors
                self.refit(new_internal);
            }
        }
    }

    /// Return all pairs of leaf collider indices whose AABBs overlap.
    pub fn query_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        // Collect all leaves
        let leaves: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.collider_index.map(|_| i))
            .collect();

        for i in 0..leaves.len() {
            for j in (i + 1)..leaves.len() {
                let ni = &self.nodes[leaves[i]];
                let nj = &self.nodes[leaves[j]];
                if ni.aabb.intersects(nj.aabb) {
                    pairs.push((
                        ni.collider_index.unwrap(),
                        nj.collider_index.unwrap(),
                    ));
                }
            }
        }
        pairs
    }

    // -----------------------------------------------------------------------
    // Internal helpers

    fn find_best_sibling(&self, _leaf: usize, root: usize) -> usize {
        // Simple greedy: walk down choosing the child that minimises combined AABB surface area
        let leaf_aabb = self.nodes[_leaf].aabb;
        let mut best = root;
        let mut best_cost = self.nodes[root].aabb.expand(leaf_aabb).surface_area();
        let mut stack = vec![root];

        while let Some(idx) = stack.pop() {
            let combined = self.nodes[idx].aabb.expand(leaf_aabb).surface_area();
            if combined < best_cost {
                best_cost = combined;
                best = idx;
            }
            if self.nodes[idx].collider_index.is_none() {
                // Internal node — check children
                if let Some(l) = self.nodes[idx].left {
                    stack.push(l);
                }
                if let Some(r) = self.nodes[idx].right {
                    stack.push(r);
                }
            }
        }
        best
    }

    fn refit(&mut self, mut idx: usize) {
        loop {
            let (left, right) = (self.nodes[idx].left, self.nodes[idx].right);
            if let (Some(l), Some(r)) = (left, right) {
                self.nodes[idx].aabb = self.nodes[l].aabb.expand(self.nodes[r].aabb);
            }
            match self.nodes[idx].parent {
                Some(p) => idx = p,
                None => break,
            }
        }
    }
}

impl Default for AabbTree {
    fn default() -> Self {
        Self::new()
    }
}

// Surface area helper used for SAH
trait SurfaceArea {
    fn surface_area(self) -> f32;
}

impl SurfaceArea for Aabb {
    fn surface_area(self) -> f32 {
        let e = self.max.sub(self.min);
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }
}
