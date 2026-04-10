"""
Asset Descriptor Generator
Generates descriptions for 3D models, textures, and game assets
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from schemas.data_schemas import AssetDescriptor
from utils.procedural_algorithms import ProceduralNameGenerator, ProceduralDescriptionGenerator


class AssetDescriptorGenerator:
    """Enterprise-grade asset description generation for game development"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize procedural generators
        self.name_gen = ProceduralNameGenerator(seed)
        self.desc_gen = ProceduralDescriptionGenerator(seed)

        # Asset types and their properties
        self.asset_types = {
            'weapon': {
                'subtypes': ['sword', 'axe', 'bow', 'staff', 'dagger', 'hammer', 'spear'],
                'materials': ['iron', 'steel', 'mithril', 'adamant', 'enchanted_wood', 'bone', 'crystal'],
                'styles': ['practical', 'ornate', 'barbaric', 'elven', 'dwarven', 'magical']
            },
            'armor': {
                'subtypes': ['helmet', 'chestplate', 'gauntlets', 'boots', 'shield', 'cloak'],
                'materials': ['leather', 'chainmail', 'plate', 'enchanted_cloth', 'dragon_scale', 'demon_hide'],
                'styles': ['light', 'medium', 'heavy', 'magical', 'ceremonial', 'stealth']
            },
            'prop': {
                'subtypes': ['furniture', 'decoration', 'container', 'tool', 'statue', 'portal'],
                'materials': ['wood', 'stone', 'metal', 'glass', 'cloth', 'crystal', 'bone'],
                'styles': ['rustic', 'elegant', 'industrial', 'magical', 'ancient', 'modern']
            },
            'environment': {
                'subtypes': ['building', 'terrain', 'vegetation', 'water', 'skybox', 'particle'],
                'materials': ['organic', 'mineral', 'liquid', 'energy', 'atmospheric'],
                'styles': ['natural', 'urban', 'fantasy', 'sci-fi', 'ruins', 'mystical']
            },
            'character': {
                'subtypes': ['humanoid', 'creature', 'mount', 'summon', 'construct'],
                'materials': ['flesh', 'scale', 'fur', 'metal', 'energy', 'shadow'],
                'styles': ['realistic', 'stylized', 'cartoon', 'anime', 'abstract']
            },
            'effect': {
                'subtypes': ['spell', 'explosion', 'aura', 'trail', 'impact', 'ambient'],
                'materials': ['energy', 'particles', 'light', 'sound', 'elemental'],
                'styles': ['subtle', 'dramatic', 'chaotic', 'orderly', 'ethereal']
            }
        }

        self.quality_levels = ['low', 'medium', 'high', 'ultra']
        self.texture_types = ['diffuse', 'normal', 'specular', 'emissive', 'roughness', 'metallic']
        self.lod_levels = ['high', 'medium', 'low', 'billboard']

    def generate_asset(self,
                      asset_type: str = None,
                      subtype: str = None,
                      quality: str = 'medium',
                      complexity: int = 3) -> AssetDescriptor:
        """Generate a complete asset descriptor"""

        asset = AssetDescriptor()
        asset.id = str(uuid.uuid4())

        # Select asset type and subtype
        if not asset_type:
            asset_type = self.rng.choice(list(self.asset_types.keys()))
        asset.asset_type = asset_type

        type_info = self.asset_types[asset_type]
        if not subtype:
            subtype = self.rng.choice(type_info['subtypes'])
        asset.subtype = subtype

        # Generate basic properties
        asset.name = self._generate_asset_name(asset_type, subtype)
        asset.description = self._generate_asset_description(asset_type, subtype, complexity)

        # Generate visual properties
        asset.visual_properties = self._generate_visual_properties(asset_type, subtype, quality)

        # Generate material properties
        asset.material_properties = self._generate_material_properties(asset_type, subtype, quality)

        # Generate technical specifications
        asset.technical_specs = self._generate_technical_specs(asset_type, quality)

        # Generate usage context
        asset.usage_context = self._generate_usage_context(asset_type, subtype)

        # Generate metadata
        asset.metadata = self._generate_asset_metadata(asset_type, subtype, quality, complexity)

        return asset

    def _generate_asset_name(self, asset_type: str, subtype: str) -> str:
        """Generate asset name"""

        name_templates = {
            'weapon': [
                "{material} {subtype}",
                "{style} {subtype}",
                "{quality} {subtype}",
                "The {legendary} {subtype}"
            ],
            'armor': [
                "{material} {subtype}",
                "{style} {subtype}",
                "{quality} {subtype}",
                "{faction} {subtype}"
            ],
            'prop': [
                "{material} {subtype}",
                "{style} {subtype}",
                "{era} {subtype}",
                "Mysterious {subtype}"
            ],
            'environment': [
                "{terrain} {subtype}",
                "{climate} {subtype}",
                "{era} {subtype}",
                "Ancient {subtype}"
            ],
            'character': [
                "{species} {subtype}",
                "{class} {subtype}",
                "{legendary} {subtype}",
                "Shadow {subtype}"
            ],
            'effect': [
                "{element} {subtype}",
                "{intensity} {subtype}",
                "Arcane {subtype}",
                "Primal {subtype}"
            ]
        }

        templates = name_templates.get(asset_type, ["{subtype}"])
        template = self.rng.choice(templates)

        # Fill template variables
        variables = {
            'material': self.rng.choice(['Iron', 'Steel', 'Mithril', 'Adamant', 'Enchanted', 'Crystal', 'Bone']),
            'style': self.rng.choice(['Elven', 'Dwarven', 'Orcish', 'Magical', 'Barbaric', 'Noble', 'Mystic']),
            'quality': self.rng.choice(['Masterwork', 'Legendary', 'Cursed', 'Blessed', 'Ancient', 'Forged']),
            'legendary': self.rng.choice(['Legendary', 'Mythical', 'Epic', 'Heroic', 'Divine', 'Infernal']),
            'faction': self.rng.choice(['Imperial', 'Tribal', 'Elven', 'Dwarven', 'Orc', 'Undead', 'Demonic']),
            'era': self.rng.choice(['Ancient', 'Medieval', 'Modern', 'Futuristic', 'Tribal', 'Victorian']),
            'terrain': self.rng.choice(['Mountain', 'Forest', 'Desert', 'Swamp', 'Tundra', 'Volcanic']),
            'climate': self.rng.choice(['Arid', 'Tropical', 'Frozen', 'Temperate', 'Barren', 'Lush']),
            'species': self.rng.choice(['Human', 'Elven', 'Dwarven', 'Orcish', 'Draconic', 'Fey', 'Undead']),
            'class': self.rng.choice(['Warrior', 'Mage', 'Rogue', 'Priest', 'Knight', 'Assassin', 'Summoner']),
            'element': self.rng.choice(['Fire', 'Ice', 'Lightning', 'Shadow', 'Light', 'Earth', 'Wind']),
            'intensity': self.rng.choice(['Gentle', 'Fierce', 'Calm', 'Violent', 'Subtle', 'Overwhelming']),
            'subtype': subtype.replace('_', ' ').title()
        }

        name = template
        for var, options in variables.items():
            if f"{{{var}}}" in name:
                name = name.replace(f"{{{var}}}", str(options))

        return name

    def _generate_asset_description(self, asset_type: str, subtype: str, complexity: int) -> str:
        """Generate detailed asset description"""

        return self.desc_gen.generate_description(
            theme=f"asset_{asset_type}_{subtype}",
            complexity=complexity,
            length='medium'
        )

    def _generate_visual_properties(self, asset_type: str, subtype: str, quality: str) -> Dict[str, Any]:
        """Generate visual properties"""

        colors = ['red', 'blue', 'green', 'gold', 'silver', 'black', 'white', 'brown', 'purple', 'orange']
        styles = ['realistic', 'stylized', 'cartoonish', 'abstract', 'ornate', 'minimalist']

        visual_props = {
            'primary_color': self.rng.choice(colors),
            'secondary_color': self.rng.choice(colors),
            'style': self.rng.choice(styles),
            'detail_level': quality,
            'texture_resolution': self._get_texture_resolution(quality),
            'normal_mapping': quality in ['high', 'ultra'],
            'specular_mapping': quality in ['medium', 'high', 'ultra'],
            'emissive': self.rng.random() > 0.7,
            'transparency': self.rng.random() > 0.8,
            'animation': self.rng.random() > 0.6
        }

        # Asset type specific properties
        if asset_type == 'effect':
            visual_props['particle_count'] = self.rng.randint(10, 1000)
            visual_props['lifetime'] = self.rng.uniform(0.5, 10.0)
        elif asset_type == 'environment':
            visual_props['tileable'] = True
            visual_props['seamless'] = True
        elif asset_type == 'character':
            visual_props['rigged'] = True
            visual_props['facial_expressions'] = quality in ['high', 'ultra']

        return visual_props

    def _get_texture_resolution(self, quality: str) -> str:
        """Get texture resolution based on quality"""

        resolutions = {
            'low': '512x512',
            'medium': '1024x1024',
            'high': '2048x2048',
            'ultra': '4096x4096'
        }

        return resolutions.get(quality, '1024x1024')

    def _generate_material_properties(self, asset_type: str, subtype: str, quality: str) -> Dict[str, Any]:
        """Generate material properties"""

        materials = self.asset_types[asset_type]['materials']
        material = self.rng.choice(materials)

        material_props = {
            'material_type': material,
            'durability': self._calculate_durability(material, quality),
            'weight': self._calculate_weight(material, asset_type, subtype),
            'rarity': self._calculate_rarity(material, quality),
            'magical_properties': self.rng.random() > 0.8,
            'degradation_rate': self.rng.uniform(0.0, 0.1)
        }

        # Add material-specific properties
        if 'metal' in material.lower():
            material_props['conductivity'] = self.rng.uniform(0.1, 1.0)
            material_props['magnetic'] = self.rng.random() > 0.5
        elif 'wood' in material.lower():
            material_props['flammability'] = self.rng.uniform(0.1, 0.8)
            material_props['flexibility'] = self.rng.uniform(0.2, 0.9)
        elif 'crystal' in material.lower():
            material_props['refractive_index'] = self.rng.uniform(1.3, 2.5)
            material_props['energy_capacity'] = self.rng.randint(10, 100)

        return material_props

    def _calculate_durability(self, material: str, quality: str) -> int:
        """Calculate material durability"""

        base_durability = {
            'wood': 30, 'leather': 25, 'cloth': 15, 'bone': 35,
            'stone': 80, 'iron': 60, 'steel': 75, 'mithril': 90,
            'adamant': 95, 'crystal': 70, 'enchanted': 85
        }

        quality_multipliers = {
            'low': 0.5, 'medium': 1.0, 'high': 1.5, 'ultra': 2.0
        }

        base = base_durability.get(material, 50)
        multiplier = quality_multipliers.get(quality, 1.0)

        return int(base * multiplier)

    def _calculate_weight(self, material: str, asset_type: str, subtype: str) -> float:
        """Calculate asset weight"""

        material_density = {
            'wood': 0.6, 'leather': 0.9, 'cloth': 0.3, 'bone': 1.8,
            'stone': 2.5, 'iron': 7.8, 'steel': 7.8, 'mithril': 5.0,
            'adamant': 8.5, 'crystal': 2.6, 'enchanted': 1.0
        }

        # Base volumes by asset type
        base_volumes = {
            'weapon': 0.01, 'armor': 0.05, 'prop': 0.1,
            'environment': 1.0, 'character': 0.08, 'effect': 0.001
        }

        density = material_density.get(material, 1.0)
        volume = base_volumes.get(asset_type, 0.1)

        # Add randomization
        volume *= self.rng.uniform(0.8, 1.2)

        return round(volume * density, 2)

    def _calculate_rarity(self, material: str, quality: str) -> str:
        """Calculate material rarity"""

        material_rarity = {
            'wood': 'common', 'leather': 'common', 'cloth': 'common', 'bone': 'uncommon',
            'stone': 'common', 'iron': 'common', 'steel': 'uncommon', 'mithril': 'rare',
            'adamant': 'epic', 'crystal': 'rare', 'enchanted': 'legendary'
        }

        quality_rarity = {
            'low': 'common', 'medium': 'uncommon', 'high': 'rare', 'ultra': 'epic'
        }

        mat_rarity = material_rarity.get(material, 'common')
        qual_rarity = quality_rarity.get(quality, 'common')

        # Combine rarities (take the rarer one)
        rarity_hierarchy = ['common', 'uncommon', 'rare', 'epic', 'legendary']
        mat_index = rarity_hierarchy.index(mat_rarity)
        qual_index = rarity_hierarchy.index(qual_rarity)

        return rarity_hierarchy[max(mat_index, qual_index)]

    def _generate_technical_specs(self, asset_type: str, quality: str) -> Dict[str, Any]:
        """Generate technical specifications"""

        specs = {
            'polygon_count': self._get_polygon_count(asset_type, quality),
            'texture_count': len(self.texture_types),
            'lod_levels': self.lod_levels[:self._get_lod_count(quality)],
            'animation_clips': self._get_animation_count(asset_type),
            'physics_enabled': self.rng.random() > 0.5,
            'collision_mesh': True,
            'optimization_level': quality
        }

        # Asset type specific specs
        if asset_type == 'character':
            specs['bone_count'] = self.rng.randint(20, 100)
            specs['facial_blendshapes'] = quality in ['high', 'ultra']
        elif asset_type == 'effect':
            specs['particle_systems'] = self.rng.randint(1, 5)
            specs['shader_complexity'] = quality
        elif asset_type == 'environment':
            specs['instancing_support'] = True
            specs['level_of_detail'] = True

        return specs

    def _get_polygon_count(self, asset_type: str, quality: str) -> int:
        """Get polygon count based on asset type and quality"""

        base_counts = {
            'weapon': 500, 'armor': 1000, 'prop': 2000,
            'environment': 5000, 'character': 8000, 'effect': 100
        }

        quality_multipliers = {
            'low': 0.25, 'medium': 0.5, 'high': 1.0, 'ultra': 2.0
        }

        base = base_counts.get(asset_type, 1000)
        multiplier = quality_multipliers.get(quality, 1.0)

        return int(base * multiplier)

    def _get_lod_count(self, quality: str) -> int:
        """Get number of LOD levels"""

        lod_counts = {
            'low': 1, 'medium': 2, 'high': 3, 'ultra': 4
        }

        return lod_counts.get(quality, 2)

    def _get_animation_count(self, asset_type: str) -> int:
        """Get number of animations"""

        if asset_type == 'character':
            return self.rng.randint(5, 20)
        elif asset_type in ['weapon', 'armor']:
            return self.rng.randint(1, 5)
        else:
            return 0

    def _generate_usage_context(self, asset_type: str, subtype: str) -> Dict[str, Any]:
        """Generate usage context"""

        contexts = {
            'weapon': ['combat', 'ceremonial', 'hunting', 'self-defense'],
            'armor': ['combat', 'ceremonial', 'work', 'travel'],
            'prop': ['decoration', 'functional', 'interactive', 'atmospheric'],
            'environment': ['background', 'interactive', 'atmospheric', 'navigational'],
            'character': ['player', 'npc', 'enemy', 'ally', 'summon'],
            'effect': ['combat', 'magic', 'environmental', 'interface']
        }

        return {
            'primary_use': self.rng.choice(contexts.get(asset_type, ['general'])),
            'secondary_uses': self.rng.sample(contexts.get(asset_type, ['general']), 2),
            'restrictions': self._generate_restrictions(asset_type),
            'compatibility': self._generate_compatibility(asset_type)
        }

    def _generate_restrictions(self, asset_type: str) -> List[str]:
        """Generate usage restrictions"""

        restrictions = []

        if asset_type in ['weapon', 'armor']:
            restrictions.extend(['level_requirement', 'class_restriction', 'alignment_restriction'])

        if asset_type == 'character':
            restrictions.extend(['faction_restriction', 'reputation_requirement'])

        # Randomly select some restrictions
        if restrictions:
            num_restrictions = self.rng.randint(0, len(restrictions))
            restrictions = self.rng.sample(restrictions, num_restrictions)

        return restrictions

    def _generate_compatibility(self, asset_type: str) -> List[str]:
        """Generate compatibility information"""

        compatibilities = {
            'weapon': ['strength', 'dexterity', 'one-handed', 'two-handed'],
            'armor': ['light', 'medium', 'heavy', 'magical'],
            'prop': ['indoor', 'outdoor', 'dungeon', 'urban'],
            'environment': ['forest', 'desert', 'mountain', 'urban'],
            'character': ['human', 'elf', 'dwarf', 'orc', 'undead'],
            'effect': ['fire', 'ice', 'lightning', 'healing']
        }

        return compatibilities.get(asset_type, ['universal'])

    def _generate_asset_metadata(self, asset_type: str, subtype: str, quality: str, complexity: int) -> Dict[str, Any]:
        """Generate asset metadata"""

        return {
            'asset_type': asset_type,
            'subtype': subtype,
            'quality': quality,
            'complexity': complexity,
            'creation_date': '2024-01-01',  # Would be dynamic in real implementation
            'artist': f"Artist_{self.rng.randint(1, 100)}",
            'license': 'proprietary',
            'tags': self._generate_tags(asset_type, subtype),
            'dependencies': self._generate_dependencies(asset_type),
            'performance_rating': self._calculate_performance_rating(quality, complexity)
        }

    def _generate_tags(self, asset_type: str, subtype: str) -> List[str]:
        """Generate asset tags"""

        base_tags = [asset_type, subtype]

        additional_tags = {
            'weapon': ['combat', 'equipment', 'damage'],
            'armor': ['protection', 'equipment', 'defense'],
            'prop': ['interactive', 'decoration', 'utility'],
            'environment': ['world', 'atmosphere', 'navigation'],
            'character': ['npc', 'enemy', 'animation'],
            'effect': ['visual', 'magic', 'feedback']
        }

        tags = base_tags + additional_tags.get(asset_type, [])
        return list(set(tags))  # Remove duplicates

    def _generate_dependencies(self, asset_type: str) -> List[str]:
        """Generate asset dependencies"""

        dependencies = []

        if asset_type == 'character':
            dependencies.extend(['skeleton_rig', 'animation_controller'])
        elif asset_type == 'effect':
            dependencies.extend(['particle_system', 'shader'])
        elif asset_type in ['weapon', 'armor']:
            dependencies.extend(['material_library', 'texture_set'])

        # Add common dependencies
        if self.rng.random() > 0.5:
            dependencies.append('audio_system')

        return dependencies

    def _calculate_performance_rating(self, quality: str, complexity: int) -> str:
        """Calculate performance rating"""

        quality_scores = {'low': 1, 'medium': 2, 'high': 3, 'ultra': 4}
        score = quality_scores.get(quality, 2) + complexity

        if score <= 3:
            return 'excellent'
        elif score <= 5:
            return 'good'
        elif score <= 7:
            return 'fair'
        else:
            return 'poor'

    def generate_batch(self, count: int, asset_type_filter: str = None) -> List[AssetDescriptor]:
        """Generate multiple asset descriptors"""

        assets = []

        for _ in range(count):
            asset_type = asset_type_filter if asset_type_filter else None
            quality = self.rng.choice(self.quality_levels)
            complexity = self.rng.randint(1, 5)

            asset = self.generate_asset(
                asset_type=asset_type,
                quality=quality,
                complexity=complexity
            )
            assets.append(asset)

        return assets

    def generate_asset_pack(self, theme: str, num_assets: int = 10) -> Dict[str, List[AssetDescriptor]]:
        """Generate a themed asset pack"""

        theme_mappings = {
            'fantasy': ['weapon', 'armor', 'character', 'prop'],
            'sci-fi': ['weapon', 'armor', 'character', 'effect'],
            'horror': ['character', 'effect', 'prop', 'environment'],
            'western': ['weapon', 'armor', 'character', 'prop'],
            'medieval': ['weapon', 'armor', 'character', 'environment']
        }

        asset_types = theme_mappings.get(theme, list(self.asset_types.keys()))

        asset_pack = {}

        for asset_type in asset_types:
            assets = self.generate_batch(
                num_assets // len(asset_types),
                asset_type_filter=asset_type
            )
            asset_pack[asset_type] = assets

        return asset_pack