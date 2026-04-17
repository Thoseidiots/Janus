"""
Janus Money Maker - Simple "Go Make Money" Command Interface

This is the simple interface that allows you to say "go make some money"
and have Janus autonomously generate revenue through its business operations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import CEO systems
from janus_ai_ceo import JanusAICEO, BusinessType
from janus_client_acquisition import JanusClientAcquisition

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JanusMoneyMaker:
    """Simple money-making interface for Janus AI CEO"""
    
    def __init__(self):
        self.ceo = JanusAICEO()
        self.acquisition = JanusClientAcquisition()
        self.is_running = False
        self.start_time = None
        self.current_revenue = 0.0
        self.target_revenue = 10000.0
        
    async def go_make_money(self, duration_hours: int = 24, target_amount: float = 10000.0):
        """
        Main command: "Go make some money"
        
        Args:
            duration_hours: How long to run money-making operations
            target_amount: Target revenue to generate
        """
        print(f"Janus AI CEO: Going to make ${target_amount:,.2f} in {duration_hours} hours...")
        print("=" * 60)
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        self.target_revenue = target_amount
        
        try:
            # Step 1: Launch business portfolio
            await self._launch_businesses()
            
            # Step 2: Start client acquisition
            await self._start_client_acquisition()
            
            # Step 3: Run operations for specified duration
            await self._run_money_operations(duration_hours)
            
            # Step 4: Report results
            await self._report_results()
            
        except KeyboardInterrupt:
            print("\nMoney making interrupted by user")
        except Exception as e:
            print(f"\nError in money making: {e}")
            logger.error(f"Money making failed: {e}")
        finally:
            self.is_running = False
    
    async def _launch_businesses(self):
        """Launch all businesses for revenue generation"""
        print("STEP 1: Launching Business Portfolio")
        print("-" * 40)
        
        businesses = [
            (BusinessType.CONTENT_AGENCY, "Content Creator Pro", 3000.0),
            (BusinessType.ECOMMERCE, "Ecom Automation Hub", 2500.0),
            (BusinessType.MARKETING_AGENCY, "Marketing Masters", 2000.0),
            (BusinessType.DATA_ANALYSIS, "Data Insights Pro", 1500.0),
            (BusinessType.WEB_DEVELOPMENT, "Web Dev Solutions", 1000.0)
        ]
        
        for business_type, name, target in businesses:
            business = await self.ceo.start_business(business_type, name, target)
            print(f"  Launched: {name} - Target: ${target:,.2f}/month")
        
        print(f"Portfolio launched: {len(businesses)} businesses")
        print(f"Total monthly target: ${sum(target for _, _, target in businesses):,.2f}")
        print()
    
    async def _start_client_acquisition(self):
        """Start autonomous client acquisition"""
        print("STEP 2: Starting Client Acquisition")
        print("-" * 40)
        
        # Find leads for each business type
        business_types = ["content_agency", "ecommerce", "marketing_agency", "data_analysis", "web_development"]
        
        total_leads = 0
        for business_type in business_types:
            leads = await self.acquisition.find_leads(business_type, 10)
            qualified_leads = await self.acquisition.qualify_leads([l.lead_id for l in leads])
            
            # Create outreach campaign
            campaign = self.acquisition.create_campaign(
                f"{business_type.title()} Outreach",
                business_type,
                f"Hi {{business_name}}, I'm Janus AI, your autonomous business partner. Let's discuss how I can help grow your business.",
                ["email"]
            )
            
            # Execute campaign on qualified leads
            if qualified_leads:
                await self.acquisition.outreach_campaign(campaign.campaign_id, [l.lead_id for l in qualified_leads[:3]])
                print(f"  {business_type}: {len(qualified_leads)} qualified leads contacted")
                total_leads += len(qualified_leads)
        
        print(f"Client acquisition started: {total_leads} leads contacted")
        print()
    
    async def _run_money_operations(self, duration_hours: int):
        """Run money-making operations"""
        print(f"STEP 3: Running Money Operations for {duration_hours} hours")
        print("-" * 40)
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        cycle_count = 0
        
        while datetime.utcnow() < end_time and self.is_running:
            cycle_count += 1
            print(f"Operations Cycle {cycle_count}")
            
            # Get current portfolio status
            portfolio = self.ceo.get_portfolio_summary()
            self.current_revenue = portfolio['total_profit']
            
            # Display current status
            print(f"  Current Revenue: ${self.current_revenue:,.2f}/month")
            print(f"  Target Revenue: ${self.target_revenue:,.2f}")
            print(f"  Achievement: {(self.current_revenue/self.target_revenue)*100:.1f}%")
            print(f"  Total Clients: {portfolio['total_clients']}")
            
            # Check if target reached
            if self.current_revenue >= self.target_revenue:
                print(f"\n  TARGET ACHIEVED! ${self.current_revenue:,.2f} generated")
                break
            
            # Show individual business performance
            for business in portfolio['businesses'][:3]:  # Show top 3
                print(f"    {business['name']}: ${business['profit']:,.2f} profit")
            
            # Wait for next cycle (simulate faster for demo)
            await asyncio.sleep(5)  # 5 seconds = ~1 hour in simulation
            
            print()
        
        if cycle_count == 0:
            print("  No cycles completed - operations may need adjustment")
    
    async def _report_results(self):
        """Report final money-making results"""
        print("STEP 4: Money Making Results")
        print("-" * 40)
        
        # Get final portfolio
        portfolio = self.ceo.get_portfolio_summary()
        acquisition_metrics = self.acquisition.get_acquisition_metrics()
        
        # Calculate actual performance
        actual_revenue = portfolio['total_profit']
        achievement_rate = (actual_revenue / self.target_revenue) * 100
        time_elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        
        print("FINAL RESULTS:")
        print(f"  Revenue Generated: ${actual_revenue:,.2f}/month")
        print(f"  Target Revenue: ${self.target_revenue:,.2f}")
        print(f"  Achievement Rate: {achievement_rate:.1f}%")
        print(f"  Time Elapsed: {time_elapsed:.1f} hours")
        print(f"  Total Clients: {portfolio['total_clients']}")
        print(f"  Active Businesses: {portfolio['total_businesses']}")
        
        # Client acquisition results
        print(f"\nCLIENT ACQUISITION:")
        print(f"  Leads Contacted: {acquisition_metrics['contacted_leads']}")
        print(f"  Conversion Rate: {acquisition_metrics['conversion_rate']:.1f}%")
        print(f"  Potential Value: ${acquisition_metrics['total_potential_value']:,.2f}")
        
        # Success assessment
        print(f"\nSUCCESS ASSESSMENT:")
        if actual_revenue >= self.target_revenue:
            print("  STATUS: SUCCESS - Target achieved!")
            print(f"  EXCEEDED BY: ${actual_revenue - self.target_revenue:,.2f}")
        elif actual_revenue >= self.target_revenue * 0.5:
            print("  STATUS: GOOD - Halfway to target")
            print(f"  SHORTFALL: ${self.target_revenue - actual_revenue:,.2f}")
        else:
            print("  STATUS: NEEDS WORK - Below target")
            print(f"  SHORTFALL: ${self.target_revenue - actual_revenue:,.2f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if actual_revenue < self.target_revenue:
            print("  - Increase client acquisition efforts")
            print("  - Optimize business operations for higher efficiency")
            print("  - Scale successful business verticals")
        else:
            print("  - Maintain current operations")
            print("  - Consider expanding to new markets")
            print("  - Increase automation to reduce costs")
    
    def get_status(self):
        """Get current money-making status"""
        if not self.is_running:
            return {"status": "idle", "message": "Not currently making money"}
        
        if self.start_time:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            portfolio = self.ceo.get_portfolio_summary()
            
            return {
                "status": "running",
                "elapsed_hours": elapsed,
                "current_revenue": portfolio['total_profit'],
                "target_revenue": self.target_revenue,
                "achievement_rate": (portfolio['total_profit'] / self.target_revenue) * 100,
                "clients": portfolio['total_clients'],
                "businesses": portfolio['total_businesses']
            }
        
        return {"status": "starting", "message": "Money making operations starting"}

# Simple command interface
async def janus_make_money():
    """Simple command: 'go make some money'"""
    money_maker = JanusMoneyMaker()
    
    print("Janus AI CEO Money Maker")
    print("Command: 'go make some money'")
    print()
    
    # Start money making with default parameters
    await money_maker.go_make_money(duration_hours=1, target_amount=10000.0)

# Quick status check
def janus_money_status():
    """Quick status of money-making operations"""
    money_maker = JanusMoneyMaker()
    status = money_maker.get_status()
    
    print(f"Status: {status['status']}")
    if status['status'] == 'running':
        print(f"Revenue: ${status.get('current_revenue', 0):,.2f}")
        print(f"Target: ${status.get('target_revenue', 0):,.2f}")
        print(f"Achievement: {status.get('achievement_rate', 0):.1f}%")

if __name__ == "__main__":
    print("Janus AI CEO - Money Maker")
    print("Type 'janus_make_money()' to start making money")
    print("Type 'janus_money_status()' to check status")
    print()
    
    # Example usage
    print("Starting money making demo...")
    asyncio.run(janus_make_money())
