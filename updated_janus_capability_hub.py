# Hub auto-loader logic
def load_autonomy_hub(hub_instance):
    cap = AutonomyCapability()
    hub_instance.register("os_control", cap.execute_action)
    print("Janus Autonomy Hub Active.")
