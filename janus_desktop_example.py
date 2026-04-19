"""
janus_desktop_example.py
========================
Example showing how Janus uses desktop interaction to work like a human.

This demonstrates Janus:
1. Opening a browser
2. Navigating to job sites
3. Taking screenshots to see what's on screen
4. Opening applications
5. Searching for and opening installed apps
"""

import asyncio
import time
from janus_autonomous_worker import JanusAutonomousWorker


async def example_browser_workflow():
    """Example: Janus opens browser and navigates to job sites"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Browser Workflow")
    print("="*60 + "\n")
    
    janus = JanusAutonomousWorker()
    
    print("Step 1: Opening Upwork in browser...")
    janus.open_upwork_browser()
    time.sleep(3)
    
    print("Step 2: Taking screenshot to see what's on screen...")
    screenshot = janus.take_screenshot()
    if screenshot:
        print(f"Screenshot saved to: {screenshot}")
    
    print("Step 3: Opening Fiverr in browser...")
    janus.open_fiverr_browser()
    time.sleep(3)
    
    print("Step 4: Taking another screenshot...")
    screenshot = janus.take_screenshot()
    if screenshot:
        print(f"Screenshot saved to: {screenshot}")
    
    print("\nBrowser workflow complete!")


async def example_app_discovery():
    """Example: Janus discovers and opens installed applications"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Application Discovery")
    print("="*60 + "\n")
    
    janus = JanusAutonomousWorker()
    
    print("Step 1: Listing all installed applications...")
    apps = janus.list_installed_apps()
    print(f"Found {len(apps)} installed applications")
    print("First 20 apps:")
    for app in apps[:20]:
        print(f"  - {app}")
    
    print("\nStep 2: Searching for specific apps...")
    
    # Search for games
    games = janus.search_apps("game")
    if games:
        print(f"Found {len(games)} games:")
        for game in games[:5]:
            print(f"  - {game}")
    
    # Search for browsers
    browsers = janus.search_apps("chrome")
    if browsers:
        print(f"Found {len(browsers)} Chrome-related apps:")
        for browser in browsers:
            print(f"  - {browser}")
    
    # Search for communication apps
    comms = janus.search_apps("discord")
    if comms:
        print(f"Found {len(comms)} Discord-related apps:")
        for comm in comms:
            print(f"  - {comm}")


async def example_open_applications():
    """Example: Janus opens various applications"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Opening Applications")
    print("="*60 + "\n")
    
    janus = JanusAutonomousWorker()
    
    print("Step 1: Opening file explorer...")
    janus.open_file_explorer()
    time.sleep(2)
    
    print("Step 2: Opening terminal...")
    janus.open_terminal()
    time.sleep(2)
    
    print("Step 3: Opening YouTube...")
    janus.open_youtube_browser()
    time.sleep(2)
    
    print("Step 4: Taking screenshot...")
    screenshot = janus.take_screenshot()
    if screenshot:
        print(f"Screenshot saved to: {screenshot}")
    
    print("\nClosing all applications...")
    janus.close_all_apps()
    print("Done!")


async def example_autonomous_workflow():
    """Example: Janus autonomous workflow"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Autonomous Workflow")
    print("="*60 + "\n")
    
    janus = JanusAutonomousWorker()
    
    print("Janus Status:")
    status = janus.get_status()
    print(f"  Name: {status['name']}")
    print(f"  Desktop Enabled: {status['desktop_enabled']}")
    print(f"  Skills: {len(status['skills'])}")
    print(f"  Jobs Claimed: {status['jobs_claimed']}")
    print(f"  Jobs Completed: {status['jobs_completed']}")
    print(f"  Current Balance: ${status['finances']['current_balance']:.2f}")
    
    print("\nAutonomous Workflow:")
    print("1. Opening Upwork to find jobs...")
    janus.open_upwork_browser()
    time.sleep(2)
    
    print("2. Taking screenshot to see available jobs...")
    screenshot = janus.take_screenshot()
    if screenshot:
        print(f"   Screenshot: {screenshot}")
    
    print("3. Opening YouTube to learn new skills...")
    janus.open_youtube_browser()
    time.sleep(2)
    
    print("4. Taking screenshot to see learning resources...")
    screenshot = janus.take_screenshot()
    if screenshot:
        print(f"   Screenshot: {screenshot}")
    
    print("5. Closing applications...")
    janus.close_all_apps()
    
    print("\nAutonomous workflow complete!")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("JANUS AUTONOMOUS WORKER - DESKTOP INTERACTION EXAMPLES")
    print("="*70)
    
    print("\nThese examples show how Janus can:")
    print("✓ Open browsers and navigate to websites")
    print("✓ Take screenshots to see what's on screen")
    print("✓ Discover and open installed applications")
    print("✓ Open file explorer and terminal")
    print("✓ Work autonomously like a human on the desktop")
    
    # Run examples
    await example_browser_workflow()
    time.sleep(2)
    
    await example_app_discovery()
    time.sleep(2)
    
    await example_open_applications()
    time.sleep(2)
    
    await example_autonomous_workflow()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)
    print("\nJanus can now:")
    print("✓ Open any website in any browser")
    print("✓ Open any installed application")
    print("✓ Take screenshots to see the desktop")
    print("✓ Search for and discover applications")
    print("✓ Interact with the desktop like a human")
    print("\nNo API keys needed - Janus truly lives on your desktop!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
