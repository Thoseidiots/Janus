scene "arania_world" version 1

  # -- Arania character ------------------------------------------------------
  entity 1
    transform position 0.0 0.0 0.0 rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    mesh_renderer mesh "assets/arania_body.obj" material "assets/arania_body.mtl"
    animator controller "arania_controller" idle_anim "idle" walk_anim "walk"
    character_controller speed 1.4 turn_speed 3.0 nav_mode "screen_edge"

  # -- Desktop background plane -----------------------------------------------
  entity 2
    transform position 0.0 0.0 -5.0 rotation 0.0 0.0 0.0 1.0 scale 16.0 9.0 1.0
    mesh_renderer mesh "builtin/quad" material "builtin/desktop_capture"
    desktop_capture update_hz 30

  # -- Lights (from Arania.mat) -----------------------------------------------
  entity 3
    transform position 4.0 8.0 5.0 rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    directional_light color 1.000 0.941 0.816 intensity 2.2

  entity 4
    transform position -4.0 2.0 -3.0 rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    point_light color 0.376 0.251 0.753 intensity 0.45 radius 20.0

  entity 5
    transform position 0.0 -2.0 3.0 rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    point_light color 0.784 0.565 0.314 intensity 0.35 radius 14.0

  # -- Navigation waypoints (screen edges) -----------------------------------
  waypoints
    point 0  -6.0 0.0 0.0
    point 1  -6.0 0.0 4.0
    point 2   0.0 0.0 4.0
    point 3   6.0 0.0 4.0
    point 4   6.0 0.0 0.0
    point 5   6.0 0.0 -4.0
    point 6   0.0 0.0 -4.0
    point 7  -6.0 0.0 -4.0
