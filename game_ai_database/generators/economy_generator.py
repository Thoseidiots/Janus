"""
Economy Generator
Generates economic systems, currencies, and trade mechanics
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from utils.procedural_algorithms import (
    ProceduralNameGenerator,
    ProceduralNumberGenerator,
    ProceduralGraphGenerator
)
from schemas.data_schemas import EconomySystem

class EconomyGenerator:
    """Generates comprehensive economic systems and trade mechanics for AAA games"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.name_gen = ProceduralNameGenerator(seed)
        self.num_gen = ProceduralNumberGenerator(seed)
        self.graph_gen = ProceduralGraphGenerator(seed)
        
        # Economic building blocks
        self.economy_types = {
            'capitalist': {
                'focus': 'profit_maximization',
                'drivers': ['supply_demand', 'competition', 'innovation'],
                'mechanisms': ['free_market', 'auctions', 'trading_posts']
            },
            'feudal': {
                'focus': 'resource_control',
                'drivers': ['land_ownership', 'loyalty', 'tradition'],
                'mechanisms': ['taxation', 'tribute', 'barter']
            },
            'magical': {
                'focus': 'arcane_value',
                'drivers': ['mana_economy', 'artifact_rarity', 'enchantment_cost'],
                'mechanisms': ['mana_crystals', 'artifact_trading', 'enchantment_services']
            },
            'post_apocalyptic': {
                'focus': 'scarcity_survival',
                'drivers': ['resource_scarcity', 'barter_systems', 'raiding'],
                'mechanisms': ['scrap_trading', 'water_economy', 'safe_zones']
            },
            'space_faring': {
                'focus': 'interstellar_trade',
                'drivers': ['fuel_costs', 'distance_multipliers', 'colonial_expansion'],
                'mechanisms': ['commodity_markets', 'mining_rights', 'transport_contracts']
            },
            'fantasy': {
                'focus': 'guild_controlled',
                'drivers': ['guild_monopolies', 'craftsmanship', 'noble_patronage'],
                'mechanisms': ['guild_halls', 'patron_systems', 'craft_licenses']
            }
        }
        
        self.currency_types = [
            'precious_metals', 'paper_money', 'digital_credits', 'mana_crystals',
            'favor_tokens', 'reputation_points', 'commodity_based', 'time_based'
        ]
        
        self.market_forces = [
            'supply_demand', 'scarcity', 'luxury_tax', 'transportation_costs',
            'production_costs', 'seasonal_variation', 'political_events', 'technological_change'
        ]
        
        self.trade_mechanisms = [
            'auctions', 'fixed_pricing', 'haggling', 'barter', 'contracts',
            'futures_trading', 'black_market', 'guild_controlled'
        ]
    
    def generate_economy(self,
                        economy_type: str = 'capitalist',
                        complexity: int = 5,
                        scale: str = 'regional') -> Dict[str, Any]:
        """Generate a complete economic system"""
        
        economy_id = str(uuid.uuid4())
        
        # Generate core economic structure
        structure = self._generate_economic_structure(economy_type, complexity, scale)
        
        # Generate currency system
        currencies = self._generate_currency_system(economy_type, complexity)
        
        # Generate market dynamics
        markets = self._generate_market_dynamics(complexity, scale)
        
        # Generate trade networks
        trade_networks = self._generate_trade_networks(scale, complexity)
        
        # Generate economic actors
        actors = self._generate_economic_actors(economy_type, complexity)
        
        # Generate resource systems
        resources = self._generate_resource_systems(complexity)
        
        # Generate economic events
        events = self._generate_economic_events(complexity)
        
        # Generate balance mechanisms
        balance = self._generate_balance_mechanisms(complexity)
        
        economy_data = {
            'id': economy_id,
            'name': self._generate_economy_name(economy_type, scale),
            'type': economy_type,
            'scale': scale,
            'complexity_rating': complexity,
            'structure': structure,
            'currencies': currencies,
            'markets': markets,
            'trade_networks': trade_networks,
            'economic_actors': actors,
            'resources': resources,
            'economic_events': events,
            'balance_mechanisms': balance,
            'estimated_stability': self._estimate_economic_stability(complexity),
            'player_impact_mechanisms': self._generate_player_impact_mechanisms(),
            'accessibility_features': self._generate_economic_accessibility_features()
        }
        
        return economy_data
    
    def _generate_economic_structure(self,
                                    economy_type: str,
                                    complexity: int,
                                    scale: str) -> Dict[str, Any]:
        """Generate the core economic structure"""
        
        type_info = self.economy_types.get(economy_type, self.economy_types['capitalist'])
        
        # Determine economic scale parameters
        scale_multipliers = {
            'local': 1.0,
            'regional': 2.0,
            'national': 3.0,
            'continental': 4.0,
            'global': 5.0,
            'interstellar': 6.0
        }
        
        scale_multiplier = scale_multipliers.get(scale, 1.0)
        
        structure = {
            'focus': type_info['focus'],
            'primary_drivers': self.rng.sample(type_info['drivers'], min(len(type_info['drivers']), 2)),
            'mechanisms': self.rng.sample(type_info['mechanisms'], min(len(type_info['mechanisms']), 3)),
            'scale_multiplier': scale_multiplier,
            'market_complexity': complexity,
            'regulation_level': self.rng.choice(['minimal', 'moderate', 'strict', 'oppressive']),
            'transparency': self.rng.choice(['transparent', 'opaque', 'corrupt', 'chaotic'])
        }
        
        # Add complexity-based features
        if complexity > 7:
            structure['advanced_mechanisms'] = self.rng.sample([
                'derivatives_trading', 'central_banking', 'insurance_markets',
                'investment_funds', 'commodity_futures', 'currency_speculation'
            ], self.rng.randint(1, 3))
        
        return structure
    
    def _generate_currency_system(self, economy_type: str, complexity: int) -> Dict[str, Any]:
        """Generate a comprehensive currency system"""
        
        # Select currency types
        currency_count = max(1, complexity // 2)
        selected_types = self.rng.sample(self.currency_types, currency_count)
        
        currencies = {}
        for currency_type in selected_types:
            currency = {
                'type': currency_type,
                'name': self._generate_currency_name(currency_type),
                'base_value': self.rng.uniform(0.01, 100.0),
                'volatility': self.rng.uniform(0.1, 1.0),
                'acceptance_rate': self.rng.uniform(0.3, 1.0),
                'issuance_mechanism': self.rng.choice(['government', 'private', 'commodity_backed', 'algorithmic']),
                'physical_properties': self._generate_currency_properties(currency_type)
            }
            currencies[currency_type] = currency
        
        # Generate exchange rates
        exchange_rates = {}
        currency_list = list(currencies.keys())
        for i, curr1 in enumerate(currency_list):
            for curr2 in currency_list[i+1:]:
                rate = self.rng.uniform(0.1, 10.0)
                exchange_rates[f"{curr1}_to_{curr2}"] = rate
                exchange_rates[f"{curr2}_to_{curr1}"] = 1.0 / rate
        
        # Generate inflation/deflation mechanics
        economic_forces = {
            'inflation_rate': self.rng.uniform(-0.05, 0.15),  # -5% to +15%
            'deflation_pressure': self.rng.uniform(0.0, 0.1),
            'velocity_multiplier': self.rng.uniform(0.8, 1.5),
            'hoarding_penalty': self.rng.uniform(0.0, 0.2)
        }
        
        return {
            'currencies': currencies,
            'exchange_rates': exchange_rates,
            'economic_forces': economic_forces,
            'conversion_fees': self._generate_conversion_fees(complexity),
            'black_market_rates': self._generate_black_market_rates(exchange_rates)
        }
    
    def _generate_market_dynamics(self, complexity: int, scale: str) -> Dict[str, Any]:
        """Generate market dynamics and forces"""
        
        # Select active market forces
        force_count = max(3, complexity)
        active_forces = self.rng.sample(self.market_forces, min(force_count, len(self.market_forces)))
        
        market_forces = {}
        for force in active_forces:
            market_forces[force] = {
                'strength': self.rng.uniform(0.1, 1.0),
                'frequency': self.rng.choice(['constant', 'seasonal', 'event_based', 'random']),
                'duration': self.rng.randint(1, 30),  # days/turns
                'impact_radius': self.rng.choice(['local', 'regional', 'global']) if scale == 'global' else 'local',
                'player_influence': self.rng.random() > 0.5
            }
        
        # Generate market segments
        market_segments = self._generate_market_segments(complexity)
        
        # Generate price volatility
        volatility_curves = {}
        for segment in market_segments:
            volatility_curves[segment] = self.num_gen.generate_curve(
                start=0.1, end=1.0, points=10, curve_type='sine_wave'
            )
        
        return {
            'active_forces': market_forces,
            'market_segments': market_segments,
            'volatility_curves': volatility_curves,
            'market_efficiency': self.rng.uniform(0.3, 0.9),
            'information_asymmetry': self.rng.uniform(0.1, 0.8)
        }
    
    def _generate_trade_networks(self, scale: str, complexity: int) -> Dict[str, Any]:
        """Generate trade networks and routes"""
        
        # Generate trade hubs
        hub_count = max(3, complexity)
        trade_hubs = []
        for i in range(hub_count):
            hub = {
                'id': str(uuid.uuid4()),
                'name': self.name_gen.generate_name('location'),
                'type': self.rng.choice(['market_town', 'trading_post', 'merchant_guild', 'black_market', 'space_station']),
                'economic_power': self.rng.uniform(0.1, 1.0),
                'specialization': self.rng.choice(['luxury_goods', 'raw_materials', 'technology', 'food', 'weapons'])
            }
            trade_hubs.append(hub)
        
        # Generate trade routes
        routes = []
        for i in range(len(trade_hubs)):
            for j in range(i+1, len(trade_hubs)):
                if self.rng.random() > 0.4:  # 60% chance of connection
                    route = {
                        'from_hub': trade_hubs[i]['id'],
                        'to_hub': trade_hubs[j]['id'],
                        'distance': self.rng.randint(10, 1000),
                        'transport_cost': self.rng.uniform(1, 100),
                        'risk_level': self.rng.uniform(0.1, 1.0),
                        'trade_volume': self.rng.uniform(0.1, 1.0),
                        'special_restrictions': self.rng.sample([
                            'toll_roads', 'pirate_infested', 'political_blockade',
                            'magical_barriers', 'weather_dependent', 'guild_controlled'
                        ], self.rng.randint(0, 2))
                    }
                    routes.append(route)
        
        # Generate trade mechanisms
        mechanisms = {}
        for mechanism in self.rng.sample(self.trade_mechanisms, min(3, len(self.trade_mechanisms))):
            mechanisms[mechanism] = {
                'efficiency': self.rng.uniform(0.5, 0.95),
                'accessibility': self.rng.uniform(0.3, 1.0),
                'profit_margin': self.rng.uniform(0.05, 0.3),
                'risk_factor': self.rng.uniform(0.0, 0.5)
            }
        
        return {
            'trade_hubs': trade_hubs,
            'trade_routes': routes,
            'trade_mechanisms': mechanisms,
            'network_density': len(routes) / max(1, len(trade_hubs) * (len(trade_hubs) - 1) / 2),
            'globalization_index': self.rng.uniform(0.1, 1.0) if scale == 'global' else 0.3
        }
    
    def _generate_economic_actors(self, economy_type: str, complexity: int) -> List[Dict]:
        """Generate economic actors and entities"""
        
        actor_types = {
            'capitalist': ['merchants', 'bankers', 'manufacturers', 'investors'],
            'feudal': ['lords', 'guild_masters', 'tax_collectors', 'land_owners'],
            'magical': ['enchanters', 'alchemists', 'artifact_dealers', 'mana_traders'],
            'post_apocalyptic': ['scavengers', 'water_merchants', 'raiders', 'settlement_leaders'],
            'space_faring': ['corporations', 'mining_syndicates', 'transport_guilds', 'colonial_administrators'],
            'fantasy': ['merchant_guilds', 'craftsmen', 'noble_traders', 'mystical_merchants']
        }
        
        selected_types = actor_types.get(economy_type, ['merchants'])
        actor_count = max(5, complexity * 2)
        
        actors = []
        for _ in range(actor_count):
            actor_type = self.rng.choice(selected_types)
            actor = {
                'id': str(uuid.uuid4()),
                'name': self.name_gen.generate_name('character'),
                'type': actor_type,
                'economic_power': self.rng.uniform(0.1, 1.0),
                'influence_radius': self.rng.choice(['local', 'regional', 'national']),
                'specialization': self.rng.choice([
                    'luxury_goods', 'bulk_trade', 'finance', 'manufacturing',
                    'resource_extraction', 'services', 'information_brokerage'
                ]),
                'personality': self.rng.choice(['greedy', 'fair', 'manipulative', 'generous', 'ruthless']),
                'goals': self.rng.sample([
                    'profit_maximization', 'market_control', 'innovation', 'stability',
                    'expansion', 'monopoly_creation', 'social_betterment'
                ], self.rng.randint(1, 3))
            }
            actors.append(actor)
        
        return actors
    
    def _generate_resource_systems(self, complexity: int) -> Dict[str, Any]:
        """Generate resource systems and commodities"""
        
        resource_categories = [
            'raw_materials', 'manufactured_goods', 'luxury_items', 'food_stuffs',
            'energy_sources', 'magical_components', 'technology', 'services'
        ]
        
        resources = {}
        resource_count = max(8, complexity * 2)
        
        for _ in range(resource_count):
            category = self.rng.choice(resource_categories)
            resource = {
                'name': self.name_gen.generate_name('item'),
                'category': category,
                'rarity': self.rng.choice(['common', 'uncommon', 'rare', 'epic', 'legendary']),
                'base_value': self.rng.uniform(1, 10000),
                'supply_elasticity': self.rng.uniform(0.1, 2.0),
                'demand_elasticity': self.rng.uniform(0.1, 2.0),
                'production_cost': self.rng.uniform(0.1, 1000),
                'transport_cost': self.rng.uniform(0.1, 100),
                'shelf_life': self.rng.randint(1, 365) if category == 'food_stuffs' else None,
                'substitutes': self.rng.sample(['none', 'partial', 'perfect'], 1)[0]
            }
            resources[resource['name']] = resource
        
        # Generate resource dependencies
        dependencies = {}
        resource_names = list(resources.keys())
        for resource_name in resource_names:
            dependency_count = self.rng.randint(0, 3)
            if dependency_count > 0:
                deps = self.rng.sample(resource_names, dependency_count)
                dependencies[resource_name] = deps
        
        return {
            'resources': resources,
            'dependencies': dependencies,
            'resource_network': self.graph_gen.generate_resource_graph(len(resources), 0.3),
            'scarcity_events': self._generate_scarcity_events(complexity)
        }
    
    def _generate_economic_events(self, complexity: int) -> List[Dict]:
        """Generate economic events and disruptions"""
        
        event_types = [
            'market_crash', 'resource_discovery', 'technological_breakthrough',
            'political_upheaval', 'natural_disaster', 'trade_war', 'monopoly_formation',
            'currency_reform', 'boom_bust_cycle', 'speculative_bubble'
        ]
        
        events = []
        event_count = max(3, complexity)
        
        for _ in range(event_count):
            event_type = self.rng.choice(event_types)
            event = {
                'type': event_type,
                'name': self._generate_event_name(event_type),
                'probability': self.rng.uniform(0.01, 0.2),
                'duration': self.rng.randint(1, 52),  # weeks
                'impact_magnitude': self.rng.uniform(0.1, 1.0),
                'affected_markets': self.rng.sample([
                    'commodities', 'currencies', 'labor', 'real_estate', 'luxury_goods'
                ], self.rng.randint(1, 3)),
                'player_intervention_possible': self.rng.random() > 0.5,
                'cascading_effects': self.rng.random() > 0.7
            }
            events.append(event)
        
        return events
    
    def _generate_balance_mechanisms(self, complexity: int) -> Dict[str, Any]:
        """Generate economic balance mechanisms"""
        
        mechanisms = {
            'wealth_redistribution': {
                'taxation_system': self.rng.choice(['progressive', 'flat', 'regressive', 'none']),
                'welfare_programs': self.rng.random() > 0.5,
                'inheritance_limits': self.rng.random() > 0.6,
                'charity_mechanisms': self.rng.random() > 0.4
            },
            'market_regulation': {
                'price_controls': self.rng.choice(['none', 'ceilings', 'floors', 'both']),
                'monopoly_busting': self.rng.random() > 0.5,
                'trade_restrictions': self.rng.random() > 0.6,
                'quality_standards': self.rng.random() > 0.7
            },
            'inflation_control': {
                'central_bank': self.rng.random() > 0.5,
                'interest_rate_policy': self.rng.choice(['fixed', 'variable', 'none']),
                'money_supply_control': self.rng.random() > 0.6,
                'reserve_requirements': self.rng.uniform(0.0, 0.5)
            },
            'social_safeguards': {
                'minimum_wage': self.rng.random() > 0.4,
                'unemployment_benefits': self.rng.random() > 0.5,
                'price_supports': self.rng.random() > 0.6,
                'subsidy_programs': self.rng.random() > 0.5
            }
        }
        
        if complexity > 7:
            mechanisms['advanced_policies'] = {
                'quantitative_easing': self.rng.random() > 0.5,
                'fiscal_stimulus': self.rng.random() > 0.6,
                'trade_agreements': self.rng.random() > 0.7,
                'economic_zones': self.rng.random() > 0.5
            }
        
        return mechanisms
    
    def _generate_currency_name(self, currency_type: str) -> str:
        """Generate currency name"""
        
        name_patterns = {
            'precious_metals': ['Gold', 'Silver', 'Platinum', 'Electrum'],
            'paper_money': ['Credits', 'Marks', 'Dollars', 'Francs'],
            'digital_credits': ['Bits', 'DataCoins', 'NetCredits', 'QuantumBits'],
            'mana_crystals': ['ManaShards', 'ArcaneCrystals', 'PowerGems', 'EnergyCores'],
            'favor_tokens': ['FavorMarks', 'DebtChits', 'ObligationTokens', 'PromiseNotes'],
            'reputation_points': ['RepPoints', 'HonorMarks', 'StatusCredits', 'RenownTokens'],
            'commodity_based': ['OilCredits', 'WaterTokens', 'FoodChits', 'MetalMarks'],
            'time_based': ['TimeCredits', 'HourTokens', 'DayMarks', 'AgeCredits']
        }
        
        options = name_patterns.get(currency_type, ['GenericCurrency'])
        return self.rng.choice(options)
    
    def _generate_currency_properties(self, currency_type: str) -> Dict[str, Any]:
        """Generate physical/digital properties of currency"""
        
        if currency_type in ['precious_metals', 'mana_crystals']:
            return {
                'form': 'physical',
                'durability': self.rng.uniform(0.7, 1.0),
                'portability': self.rng.uniform(0.3, 0.8),
                'counterfeit_difficulty': self.rng.uniform(0.5, 1.0),
                'aesthetic_value': self.rng.uniform(0.6, 1.0)
            }
        elif currency_type in ['paper_money', 'favor_tokens']:
            return {
                'form': 'physical',
                'durability': self.rng.uniform(0.2, 0.7),
                'portability': self.rng.uniform(0.8, 1.0),
                'counterfeit_difficulty': self.rng.uniform(0.3, 0.8),
                'aesthetic_value': self.rng.uniform(0.3, 0.8)
            }
        else:  # digital currencies
            return {
                'form': 'digital',
                'durability': 1.0,
                'portability': 1.0,
                'counterfeit_difficulty': self.rng.uniform(0.8, 1.0),
                'aesthetic_value': self.rng.uniform(0.1, 0.5)
            }
    
    def _generate_conversion_fees(self, complexity: int) -> Dict[str, float]:
        """Generate currency conversion fees"""
        
        return {
            'bank_conversion_fee': self.rng.uniform(0.01, 0.05),
            'merchant_fee': self.rng.uniform(0.02, 0.08),
            'black_market_premium': self.rng.uniform(0.05, 0.20),
            'international_fee': self.rng.uniform(0.03, 0.10) if complexity > 5 else 0.0
        }
    
    def _generate_black_market_rates(self, exchange_rates: Dict[str, float]) -> Dict[str, float]:
        """Generate black market exchange rates"""
        
        black_rates = {}
        for rate_key, rate in exchange_rates.items():
            # Black market rates are typically worse for buyers
            variance = self.rng.uniform(0.8, 1.5)
            black_rates[rate_key] = rate * variance
        
        return black_rates
    
    def _generate_market_segments(self, complexity: int) -> List[str]:
        """Generate market segments"""
        
        base_segments = ['luxury_goods', 'essentials', 'services', 'raw_materials']
        
        if complexity > 5:
            base_segments.extend(['technology', 'magical_artifacts', 'rare_earths', 'information'])
        
        if complexity > 8:
            base_segments.extend(['derivatives', 'futures', 'options', 'insurance'])
        
        return base_segments
    
    def _generate_scarcity_events(self, complexity: int) -> List[Dict]:
        """Generate resource scarcity events"""
        
        events = []
        event_count = max(2, complexity // 2)
        
        for _ in range(event_count):
            event = {
                'resource': self.rng.choice(['food', 'water', 'energy', 'rare_materials', 'mana']),
                'severity': self.rng.choice(['mild', 'moderate', 'severe', 'catastrophic']),
                'duration': self.rng.randint(1, 12),  # months
                'causes': self.rng.sample([
                    'natural_disaster', 'war', 'overconsumption', 'supply_disruption',
                    'technological_failure', 'political_decision', 'magical_cataclysm'
                ], self.rng.randint(1, 2)),
                'mitigation_options': self.rng.sample([
                    'import_substitutes', 'rationing', 'price_controls', 'new_technology',
                    'alternative_sources', 'conservation_efforts', 'magical_solutions'
                ], self.rng.randint(1, 3))
            }
            events.append(event)
        
        return events
    
    def _generate_event_name(self, event_type: str) -> str:
        """Generate economic event name"""
        
        name_templates = {
            'market_crash': ['The Great Crash', 'Black Monday', 'Economic Meltdown'],
            'resource_discovery': ['Golden Discovery', 'Resource Boom', 'New Frontier'],
            'technological_breakthrough': ['Innovation Revolution', 'Tech Breakthrough', 'Digital Dawn'],
            'political_upheaval': ['Regime Change', 'Political Crisis', 'Revolutionary Wave'],
            'natural_disaster': ['Catastrophic Event', 'Natural Disaster', 'Environmental Crisis'],
            'trade_war': ['Trade Conflict', 'Economic Warfare', 'Commercial Dispute'],
            'monopoly_formation': ['Market Consolidation', 'Corporate Dominance', 'Monopoly Formation'],
            'currency_reform': ['Monetary Reform', 'Currency Revolution', 'Financial Overhaul'],
            'boom_bust_cycle': ['Economic Cycle', 'Boom and Bust', 'Market Volatility'],
            'speculative_bubble': ['Speculative Mania', 'Bubble Burst', 'Investment Frenzy']
        }
        
        templates = name_templates.get(event_type, ['Economic Event'])
        return self.rng.choice(templates)
    
    def _generate_economy_name(self, economy_type: str, scale: str) -> str:
        """Generate economy name"""
        
        type_adjectives = {
            'capitalist': ['Free', 'Market', 'Capital', 'Trade'],
            'feudal': ['Feudal', 'Noble', 'Land', 'Traditional'],
            'magical': ['Arcane', 'Magical', 'Enchanted', 'Mystical'],
            'post_apocalyptic': ['Wasteland', 'Survival', 'Scavenged', 'Ruined'],
            'space_faring': ['Interstellar', 'Galactic', 'Space', 'Cosmic'],
            'fantasy': ['Guild', 'Craft', 'Mystical', 'Legendary']
        }
        
        scale_adjectives = {
            'local': ['Village', 'Town', 'Local'],
            'regional': ['Regional', 'Provincial', 'District'],
            'national': ['National', 'Kingdom', 'Empire'],
            'continental': ['Continental', 'Worldwide', 'Global'],
            'global': ['Universal', 'World', 'Global'],
            'interstellar': ['Galactic', 'Universal', 'Cosmic']
        }
        
        adj1 = self.rng.choice(type_adjectives.get(economy_type, ['Economic']))
        adj2 = self.rng.choice(scale_adjectives.get(scale, ['System']))
        
        return f"The {adj1} {adj2} Economy"
    
    def _estimate_economic_stability(self, complexity: int) -> float:
        """Estimate economic stability"""
        
        base_stability = 0.7
        complexity_penalty = (complexity - 5) * 0.05
        stability = base_stability - complexity_penalty
        
        return max(0.1, min(1.0, stability))
    
    def _generate_player_impact_mechanisms(self) -> List[Dict]:
        """Generate mechanisms for player economic impact"""
        
        mechanisms = [
            {
                'type': 'trading',
                'description': 'Buy low, sell high in various markets',
                'skill_required': 'merchant',
                'risk_level': 'medium',
                'reward_potential': 'high'
            },
            {
                'type': 'crafting',
                'description': 'Create valuable items from raw materials',
                'skill_required': 'craftsman',
                'risk_level': 'low',
                'reward_potential': 'medium'
            },
            {
                'type': 'investing',
                'description': 'Invest in businesses and ventures',
                'skill_required': 'investor',
                'risk_level': 'high',
                'reward_potential': 'very_high'
            },
            {
                'type': 'speculating',
                'description': 'Bet on market movements and events',
                'skill_required': 'gambler',
                'risk_level': 'very_high',
                'reward_potential': 'extreme'
            }
        ]
        
        return self.rng.sample(mechanisms, self.rng.randint(2, 4))
    
    def _generate_economic_accessibility_features(self) -> List[str]:
        """Generate economic accessibility features"""
        
        features = [
            'tutorial_economy',
            'starting_capital_boost',
            'forgiving_debt_system',
            'alternative_income_sources',
            'economic_assistance_npcs',
            'clear_price_display',
            'budget_planning_tools',
            'economic_difficulty_settings'
        ]
        
        return self.rng.sample(features, self.rng.randint(3, 6))
    
    def generate_batch(self,
                       count: int,
                       economy_types: List[str] = None,
                       scales: List[str] = None) -> List[Dict]:
        """Generate multiple economic systems"""
        
        if economy_types is None:
            economy_types = list(self.economy_types.keys())
        
        if scales is None:
            scales = ['local', 'regional', 'national', 'global']
        
        economies = []
        
        for i in range(count):
            economy_type = self.rng.choice(economy_types)
            scale = self.rng.choice(scales)
            complexity = self.rng.randint(3, 10)
            
            economy = self.generate_economy(
                economy_type=economy_type,
                complexity=complexity,
                scale=scale
            )
            economies.append(economy)
        
        return economies