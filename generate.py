import torch
from diffusers import SanaPAGPipeline
from pathlib import Path
import json
import time
from datetime import datetime
import logging
import random
from PIL import Image
import numpy as np
import sys
from typing import Optional

# Configuração de logging permanece a mesma
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)

class PixelArtGenerator:
    def __init__(self, output_dir="pixel_art_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.prompts_dir = self.output_dir / "prompts"  # Novo diretório para prompts
        
        # Lista para armazenar todas as imagens e labels
        self.all_images = []
        self.all_metadata = []
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Contador de imagens existentes
        self.existing_images = len(list(self.images_dir.glob("*.png")))
        logging.info(f"Found {self.existing_images} existing images")
        
        # Inicialização do modelo permanece a mesma
        self.pipe = SanaPAGPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            variant="fp16",
            torch_dtype=torch.float16,
            pag_applied_layers="transformer_blocks.8",
        )
        self.pipe.to("cuda")
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)

        # Configurações alinhadas com o LunarCoreVAE
        self.max_seq_length = 77  # Mesmo do modelo
        
        # Tokens especiais para estruturar os prompts
        self.special_tokens = {
            'style': '[STYLE]',
            'category': '[CATEGORY]',
            'detail': '[DETAIL]',
            'color': '[COLOR]',
            'end': '[END]'
        }
        
        # Pesos para seleção de categoria
        self.style_weights = {
            'animals': 0.1,      # 10% animals
            'vehicles': 0.1,     # 10% vehicles
            'forests': 0.1,      # 10% forests
            'everyday': 0.1,     # 10% everyday items
            'objects': 0.1,      # 10% objects (tools, instruments, etc.)
            'scenes': 0.1,       # 10% scenes (interiors, outdoors, etc.)
            'creatures': 0.1,    # 10% creatures (monsters, mythical, robots)
            'emotions': 0.1,     # 10% emotions (faces, emotional scenes)
            'concepts': 0.1,     # 10% concepts (abstract, symbolic)
            'styles': 0.1        # 10% specific styles (cyberpunk, steampunk, etc.)
        }
        
        # Enhanced pixel art styles with more specific techniques
        self.pixel_art_styles = [
            "classic 8-bit pixel art style with limited palette",
            "16-bit era pixel art with enhanced color depth",
            "modern pixel art with smooth gradients",
            "high-contrast pixel art with bold outlines",
            "minimalist pixel art with essential details",
            "detailed pixel art with intricate shading",
            "isometric pixel art with precise perspective",
            "dithered pixel art with texture patterns",
            "low-resolution pixel art with clear readability",
            "high-resolution pixel art with fine details",
            "retro-gaming inspired pixel art style",
            "anime-influenced pixel art aesthetic",
            "geometric pixel art with clean shapes",
            "organic pixel art with natural curves",
            "atmospheric pixel art with mood lighting"
        ]
        
        # Enhanced color styles with advanced techniques
        self.color_styles = [
            "using limited 4-color palette with strategic placement",
            "employing color ramping for smooth gradients",
            "utilizing complementary color harmony",
            "implementing split-complementary color scheme",
            "applying analogous color relationships",
            "featuring monochromatic shading techniques",
            "using triadic color harmony for vibrancy",
            "employing color indexing for efficiency",
            "utilizing hue shifting for dynamic lighting",
            "implementing custom dithering patterns",
            "using selective color emphasis",
            "applying atmospheric color grading",
            "featuring palette cycling effects",
            "employing color-based anti-aliasing",
            "using strategic color contrast"
        ]
        
        # Enhanced pixel details with advanced techniques
        self.pixel_details = [
            "using pixel-perfect outlines with clean edges",
            "employing advanced noise dithering for texture",
            "implementing selective anti-aliasing for smoothness",
            "utilizing cluster dithering for gradient effects",
            "applying pixel-level highlights and shadows",
            "featuring precise pixel clusters for detail",
            "using strategic single-pixel placement",
            "implementing advanced outline techniques",
            "employing sub-pixel animation principles",
            "utilizing pixel-perfect perspective guidelines",
            "applying advanced texture mapping techniques",
            "featuring precise pixel-level color transitions",
            "using strategic negative space in pixel placement",
            "implementing advanced pixel clustering for detail",
            "employing pixel-perfect geometric patterns"
        ]

        # Elementos de composição compartilhados
        self.perspectives = {
            'everyday': ["front view", "3/4 view", "isometric view", "top-down view", "dynamic angle", "dutch angle", "low angle", "high angle", "birds eye view", "worms eye view"],
            'futuristic': ["wide view", "focused view", "establishing shot", "detailed close-up", "extreme close-up", "panoramic sweep", "tracking shot", "orbital view", "matrix view", "quantum perspective"],
            'scifi': ["panoramic view", "dramatic angle", "atmospheric shot", "detailed view", "dimensional view", "temporal view", "parallel universe angle", "multiverse perspective", "quantum realm view", "cosmic scale shot"],
            'city': ["aerial view", "street level", "isometric cityscape", "architectural view", "urban canyon view", "rooftop perspective", "underground view", "transit level view", "district overview", "metropolitan scale"],
            'vehicle': ["front 3/4 view", "side profile", "dynamic angle", "action shot", "mechanical detail", "interior view", "blueprint perspective", "cross-section view", "technical detail", "aerodynamic angle"],
            'mythical': ["epic shot", "dramatic pose", "atmospheric scene", "detailed portrait", "legendary perspective", "mythological frame", "divine angle", "heroic stance", "ethereal view", "immortal aspect"],
            'character': ["full body shot", "portrait view", "action pose", "character study", "emotional close-up", "dynamic movement", "hero shot", "villain pose", "side profile", "dramatic entrance"],
            'food': ["overhead shot", "close-up detail", "presentation view", "artistic arrangement", "gourmet perspective", "plating detail", "ingredient focus", "culinary composition", "garnish highlight", "texture detail"]
        }

        self.lighting = {
            'everyday': ["soft natural lighting", "warm indoor light", "bright daylight", "ambient lighting"],
            'futuristic': ["neon accents", "holographic glow", "synthetic lighting", "dynamic light trails"],
            'scifi': ["alien sun", "bioluminescent", "quantum particles", "energy fields"],
            'city': ["golden hour", "neon night", "rainy atmosphere", "morning fog"],
            'vehicle': ["studio lighting", "environmental lighting", "dramatic shadows", "tech glow"],
            'mythical': ["magical glow", "ethereal light", "mystical atmosphere", "divine rays"],
            'character': ["dramatic lighting", "ambient occlusion", "rim light", "mood lighting"],
            'food': ["professional lighting", "natural window light", "mood lighting", "spotlight"]
        }

        # Enhanced lighting effects
        self.lighting_effects = [
            "dynamic pixel-perfect lighting and shadows",
            "atmospheric light scatter effects",
            "precise rim lighting highlights",
            "detailed ambient occlusion",
            "strategic specular highlights",
            "volumetric lighting effects",
            "cast shadow details",
            "environmental light bounce",
            "mood-enhancing color temperature",
            "dramatic contrast lighting",
            "subtle gradient illumination",
            "pixel-perfect light rays",
            "atmospheric fog effects",
            "strategic highlight placement",
            "depth-enhancing shadow work"
        ]

        # Generation parameters optimized for quality
        self.generation_params = {
            'animals': {
                'guidance_scale': 10.0,  # Increased for better detail
                'pag_scale': 3.5,       # Increased for better adherence to style
                'num_steps': 100,       # More steps for better quality
                'pixel_size': 3         # Smaller pixels for finer detail
            },
            'vehicles': {
                'guidance_scale': 11.0,  # Higher for mechanical precision
                'pag_scale': 4.0,       # Higher for clean lines
                'num_steps': 100,
                'pixel_size': 3
            },
            'forests': {
                'guidance_scale': 9.5,   # Lower for natural variation
                'pag_scale': 3.5,
                'num_steps': 120,        # More steps for complex scenes
                'pixel_size': 4          # Larger for natural look
            },
            'everyday': {
                'guidance_scale': 10.0,
                'pag_scale': 3.5,
                'num_steps': 90,
                'pixel_size': 3
            },
            'objects': {
                'guidance_scale': 11.0,  # Higher for clear details
                'pag_scale': 4.0,
                'num_steps': 90,
                'pixel_size': 2          # Very small for precise details
            },
            'scenes': {
                'guidance_scale': 9.0,   # Lower for artistic freedom
                'pag_scale': 3.0,
                'num_steps': 120,        # More steps for complex scenes
                'pixel_size': 4          # Larger for scene coherence
            },
            'creatures': {
                'guidance_scale': 10.5,  # Higher for creature details
                'pag_scale': 3.5,
                'num_steps': 100,
                'pixel_size': 3
            },
            'emotions': {
                'guidance_scale': 9.0,   # Lower for expressive freedom
                'pag_scale': 3.0,
                'num_steps': 90,
                'pixel_size': 3
            },
            'concepts': {
                'guidance_scale': 8.5,   # Lowest for abstract freedom
                'pag_scale': 3.0,
                'num_steps': 100,
                'pixel_size': 4          # Larger for abstract forms
            },
            'styles': {
                'guidance_scale': 10.0,
                'pag_scale': 4.0,        # Higher for style adherence
                'num_steps': 110,
                'pixel_size': 3
            }
        }

        # Cenas de ficção científica atualizadas
        self.scifi_scenes = [
            "advanced alien city", "crystalline space station", "quantum research lab",
            "terraformed mars colony", "interdimensional gateway", "biotech research facility",
            "alien artifact chamber", "cosmic anomaly site", "xenobiology habitat",
            "deep space outpost", "temporal research center", "alien ruins exploration",
            "zero-gravity garden", "plasma energy core", "dimensional nexus"
        ]
        
        # Elementos sci-fi específicos
        self.scifi_elements = [
            "exotic matter containment", "alien technology interface", "quantum field generators",
            "bio-organic structures", "energy crystallization", "gravity manipulation devices",
            "temporal distortion fields", "neural network cores", "plasma containment fields",
            "dimensional stabilizers", "xenomorphic architecture", "bio-luminescent organisms"
        ]

        # Animal types with varied styles
        self.animal_types = [
            # High-Detail Cute Style
            "perfectly crafted pixel kitty with expressive eyes and smooth fur details at 128x128",
            "meticulously designed pixel puppy with detailed fur patterns and clean outlines",
            "precision-crafted pixel bunny with fluffy details and perfect proportions",
            "carefully pixelated hamster with round features and clean edge work",
            "detailed pixel bird with perfectly defined feathers and smooth animations",
            
            # Professional Game Style
            "studio-quality pixel fox with optimal shading and clean lines",
            "high-fidelity pixel wolf with perfect muscle definition",
            "professional pixel bear with detailed fur texturing",
            "carefully crafted pixel deer with clean antler details",
            "precision-designed pixel raccoon with perfect mask patterns",
            
            # Fantasy/Mythical
            "meticulously crafted pixel dragon with detailed scales at 128x128",
            "professional pixel phoenix with perfect flame effects",
            "high-detail pixel unicorn with magical particle effects",
            "carefully designed pixel griffin with clean feather-fur transition",
            "studio-quality pixel kitsune with perfect magical details",
            
            # Special Variations
            "cyber-enhanced pixel cat with clean neon highlights",
            "steampunk-styled pixel owl with detailed mechanical parts",
            "crystal-formed pixel deer with perfect transparency effects",
            "ghost-type pixel fox with smooth ethereal effects",
            "armored pixel bear with detailed plate work"
        ]

        # Enhanced animal poses for pixel art
        self.animal_poses = [
            "in pixel-perfect profile view",
            "with dynamic pixel action pose",
            "in 3/4 pixel perspective",
            "with isometric pixel positioning",
            "in classic side-view sprite style",
            "with animated pixel movement",
            "in detailed front-facing pose",
            "with pixel-perfect proportions",
            "in dynamic jumping motion",
            "with characteristic pixel expression",
            "in resting pixel pose",
            "with active pixel animation frame",
            "in strategic pixel composition",
            "with emphasized pixel silhouette",
            "in balanced pixel stance"
        ]

        # Forest types with enhanced details
        self.forest_types = [
            # Seasonal Forests
            "meticulously crafted autumn forest with perfect leaf details",
            "high-detail spring forest with clean cherry blossom effects",
            "professional winter forest with precise snow coverage",
            "carefully designed summer forest with perfect lighting",
            
            # Specialized Types
            "precision-crafted magical forest with particle effects",
            "studio-quality crystal forest with transparency work",
            "high-fidelity ancient forest with detailed moss effects",
            "carefully designed bamboo forest with perfect stalk details",
            
            # Biome Variations
            "perfectly pixelated rainforest with layered canopy",
            "professional pine forest with detailed needle work",
            "high-detail redwood forest with perfect scaling",
            "carefully crafted mangrove forest with clean root systems",
            
            # Fantasy Types
            "meticulously designed fairy forest with glow effects",
            "precision-crafted spirit forest with ethereal details",
            "studio-quality mushroom forest with perfect caps",
            "high-fidelity crystal grove with transparency effects"
        ]

        # Forest elements with clean details
        self.forest_elements = [
            "with simple ground cover",
            "with clear streams and rocks",
            "with distinct foliage layers",
            "with bold flower patches",
            "with clean fallen logs",
            "with sharp mushroom clusters",
            "with crisp moss details",
            "with clear forest paths",
            "with simple tree spacing",
            "with distinct light rays"
        ]

        # Enhanced backgrounds with pixel art focus
        self.backgrounds = [
            "pixel-perfect gradient background",
            "retro-styled checkered pattern",
            "atmospheric pixel clouds",
            "detailed pixel landscape",
            "minimalist pixel grid",
            "parallax-ready pixel layers",
            "dithered fade background",
            "geometric pixel patterns",
            "pixel-perfect starfield",
            "textured pixel environment",
            "clean single-color backdrop",
            "dynamic scrolling background",
            "pixel-perfect mountain range",
            "atmospheric pixel cityscape",
            "detailed pixel forest scene"
        ]

        # Vehicle types with varied styles
        self.vehicle_types = [
            # Modern Vehicles
            "precision-crafted pixel sports car with perfect reflection mapping",
            "high-detail pixel motorcycle with clean chrome effects",
            "professional pixel racing car with dynamic lighting",
            "carefully designed pixel SUV with perfect panel lines",
            "studio-quality pixel truck with detailed mechanical parts",
            
            # Fantasy/Sci-fi
            "meticulously crafted hover car with clean energy effects",
            "detailed pixel spaceship with perfect thruster animations",
            "high-fidelity pixel mech with smooth joint articulation",
            "carefully designed crystal carriage with transparency effects",
            "professional pixel time machine with temporal distortion",
            
            # Classic/Retro
            "perfectly pixelated classic car with clean chrome details",
            "precision-crafted vintage motorcycle with detailed engine",
            "studio-quality steam locomotive with perfect smoke effects",
            "high-detail pixel muscle car with clean paint effects",
            "carefully designed retro scooter with perfect proportions",
            
            # Special Types
            "meticulously crafted submarine with bubble animations",
            "professional pixel helicopter with rotating blade effects",
            "high-fidelity airship with detailed propulsion systems",
            "carefully designed hover bike with energy trail effects",
            "precision-crafted battle tank with detailed treads"
        ]

        # Everyday items with enhanced quality
        self.everyday_items = [
            # Technology
            "meticulously crafted pixel laptop with perfect screen glow",
            "high-detail pixel smartphone with clean UI elements",
            "professional pixel game console with LED effects",
            "carefully designed pixel smartwatch with animated display",
            
            # Household Items
            "precision-crafted pixel teacup with perfect steam effects",
            "studio-quality pixel lamp with detailed lighting",
            "high-fidelity pixel bookshelf with clean book spines",
            "carefully designed pixel clock with smooth hands",
            
            # Magical Items
            "perfectly pixelated magic wand with particle effects",
            "professional pixel potion bottle with liquid animations",
            "high-detail pixel spell book with glowing runes",
            "carefully crafted pixel crystal ball with transparency",
            
            # Special Objects
            "meticulously designed pixel treasure chest with gold gleam",
            "precision-crafted pixel music box with animated notes",
            "studio-quality pixel terrarium with detailed plants",
            "high-fidelity pixel robot companion with LED details"
        ]

        # Enhanced everyday contexts
        self.everyday_contexts = [
            # Interior Spaces
            "in a meticulously detailed cozy room with perfect lighting",
            "within a high-fidelity modern apartment with clean lines",
            "inside a professional studio space with perfect shadows",
            "in a carefully crafted workshop with detailed tools",
            
            # Special Environments
            "within a precision-designed magical study with particle effects",
            "in a studio-quality tech lab with clean hologram details",
            "inside a high-detail steampunk workshop with gear animations",
            "within a carefully crafted crystal chamber with transparency",
            
            # Natural Settings
            "in a perfectly rendered garden with detailed plants",
            "within a professional greenhouse with clean glass effects",
            "inside a high-fidelity cave with perfect crystal formations",
            "in a carefully designed treehouse with wooden textures",
            
            # Themed Locations
            "within a meticulously crafted space station with clean tech",
            "in a precision-designed underwater dome with bubble effects",
            "inside a studio-quality cloud castle with perfect mist",
            "within a high-detail ancient ruins with weathering effects"
        ]

        # Objects with subcategories
        self.object_types = {
            'tools': [
                "meticulously detailed pixel hammer with wood grain texture",
                "high-fidelity pixel screwdriver with metallic reflections",
                "precision-crafted pixel saw with detailed teeth",
                "carefully designed pixel wrench with chrome finish",
                "professional pixel drill with mechanical details"
            ],
            'instruments': [
                "studio-quality pixel guitar with string detail",
                "high-detail pixel piano with individual keys",
                "carefully crafted pixel drum set with cymbal effects",
                "precision-designed pixel violin with wood texture",
                "professional pixel saxophone with brass reflections"
            ],
            'magic': [
                "mystical pixel wand with particle effects",
                "glowing pixel potion with liquid animations",
                "ancient pixel scroll with magical runes",
                "enchanted pixel crystal with transparency",
                "magical pixel staff with energy effects"
            ],
            'weapons': [
                "legendary pixel sword with edge highlights",
                "detailed pixel bow with string physics",
                "high-tech pixel firearm with metal finish",
                "ancient pixel dagger with ornate details",
                "magical pixel staff with energy effects"
            ],
            'jewelry': [
                "precious pixel ring with gem reflections",
                "detailed pixel necklace with chain links",
                "ornate pixel crown with jewel details",
                "delicate pixel earrings with metal shine",
                "magical pixel amulet with glowing effects"
            ]
        }

        # Scene types with enhanced variety
        self.scene_types = {
            'interiors': [
                "cozy pixel living room with detailed furniture",
                "detailed pixel bedroom with soft lighting",
                "professional pixel kitchen with appliances",
                "pixel library with packed bookshelves",
                "pixel workshop with tool arrangements"
            ],
            'outdoors': [
                "serene pixel meadow with wildflowers",
                "majestic pixel mountains with snow caps",
                "detailed pixel beach with wave animations",
                "dense pixel forest with layered trees",
                "pixel desert with sand dune patterns"
            ],
            'events': [
                "joyous pixel party with decorations",
                "elegant pixel wedding with detailed scene",
                "solemn pixel funeral with atmosphere",
                "festive pixel carnival with lights",
                "pixel concert with crowd details"
            ],
            'places': [
                "busy pixel school with classrooms",
                "detailed pixel hospital corridor",
                "charming pixel store with displays",
                "pixel restaurant with table settings",
                "pixel train station with platforms"
            ]
        }

        # Creature types with varied designs
        self.creature_types = {
            'monsters': [
                "fearsome pixel dragon with scale details",
                "creepy pixel zombie with decay effects",
                "alien pixel creature with bioluminescence",
                "giant pixel monster with fur texture",
                "pixel demon with flame effects"
            ],
            'mythical': [
                "majestic pixel unicorn with sparkle effects",
                "delicate pixel fairy with wing details",
                "graceful pixel mermaid with scale shine",
                "pixel phoenix with flame particles",
                "pixel griffin with feather details"
            ],
            'robots': [
                "advanced pixel android with LED details",
                "battle pixel cyborg with mechanical parts",
                "hovering pixel drone with propulsion",
                "pixel mech with joint articulation",
                "utility pixel robot with tool attachments"
            ]
        }

        # Emotion expressions and scenes
        self.emotion_types = {
            'faces': [
                "joyful pixel face with expressive eyes",
                "sad pixel portrait with subtle tears",
                "angry pixel expression with effects",
                "fearful pixel face with wide eyes",
                "peaceful pixel expression with soft features"
            ],
            'scenes': [
                "serene pixel landscape with calming elements",
                "intense pixel battle with action effects",
                "romantic pixel scene with mood lighting",
                "mysterious pixel setting with fog",
                "cheerful pixel celebration with details"
            ]
        }

        # Abstract and symbolic concepts
        self.concept_types = {
            'abstract': [
                "pixel representation of love with hearts",
                "peace concept in pixel art style",
                "freedom expressed through pixel birds",
                "pixel art showing the concept of time",
                "abstract pixel pattern representing growth"
            ],
            'symbolic': [
                "detailed pixel logo with clean edges",
                "pixel flag with waving animation",
                "symbolic pixel sign with clear text",
                "pixel emblem with heraldic details",
                "pixel icon with precise design"
            ]
        }

        # Specific style variations
        self.style_variations = {
            'cyberpunk': [
                "neon-lit pixel cityscape with rain",
                "cyber pixel character with augments",
                "pixel tech interface with glow",
                "dystopian pixel street with signs",
                "pixel cyberspace with grid effects"
            ],
            'steampunk': [
                "Victorian pixel scene with gears",
                "steam-powered pixel machine details",
                "pixel airship with brass fixtures",
                "steampunk pixel character with goggles",
                "pixel workshop with steam effects"
            ],
            'fantasy': [
                "magical pixel castle with effects",
                "pixel wizard tower with details",
                "fantasy pixel village with charm",
                "pixel dragon lair with treasure",
                "enchanted pixel forest with spirits"
            ],
            'scifi': [
                "futuristic pixel spaceship interior",
                "alien pixel planet landscape",
                "high-tech pixel laboratory scene",
                "pixel space station with details",
                "sci-fi pixel battle scene"
            ]
        }

    def pixelate_image(self, image, pixel_size=4):
        """Processo de pixelação melhorado para preservar mais detalhes"""
        img = np.array(image)
        height, width = img.shape[:2]
        
        # Reduz a imagem mantendo mais detalhes
        h = height // pixel_size
        w = width // pixel_size
        
        # Primeiro redimensiona com LANCZOS para preservar detalhes
        small = Image.fromarray(img).resize((w, h), Image.LANCZOS)
        
        # Quantização de cores mais suave
        small = small.quantize(colors=64, method=2).convert('RGB')  # Aumentado para 64 cores
        
        # Volta ao tamanho original com NEAREST para manter pixels nítidos
        final = small.resize((width, height), Image.NEAREST)
        
        return final

    def generate_prompt(self, category="random"):
        if category == "random":
            category = random.choices(
                list(self.style_weights.keys()),
                weights=list(self.style_weights.values())
            )[0]
        
        # Select core components
        style = random.choice(self.pixel_art_styles)
        lighting = random.choice(self.lighting_effects)
        color_style = random.choice(self.color_styles)
        pixel_detail = random.choice(self.pixel_details)
        
        # Add pixel art quality emphasis
        quality_emphasis = random.choice([
            "with emphasis on clean pixel edges and precise detail placement",
            "focusing on pixel-perfect proportions and clear silhouettes",
            "maintaining consistent pixel scale and precise spacing",
            "emphasizing crisp pixel definition and careful anti-aliasing",
            "with attention to pixel-level detail and clean transitions"
        ])
        
        # Add technical specifications
        tech_specs = "in 128x128 resolution with pixel-perfect precision"
        
        base_prompt = ""
        
        if category == "animals":
            animal = random.choice(self.animal_types)
            pose = random.choice(self.animal_poses)
            background = random.choice(self.backgrounds)
            
            base_prompt = f"{self.special_tokens['category']}animals{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {animal}, {pose}, {quality_emphasis}, {tech_specs}, with {background}, {lighting}"
            
        elif category == "vehicles":
            vehicle = random.choice(self.vehicle_types)
            environment = random.choice([
                "on a clean pixel-grid road with simple backdrop",
                "in a pixel-perfect urban setting with distinct lights",
                "on a well-defined pixel track with clean elements",
                "in a pixel art showroom with focused lighting",
                "in a distinct pixel scene with clear motion"
            ])
            
            base_prompt = f"{self.special_tokens['category']}vehicles{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {vehicle}, {environment}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "forests":
            forest = random.choice(self.forest_types)
            elements = random.choice(self.forest_elements)
            time_of_day = random.choice([
                "during pixel-perfect sunset", "at distinct pixel twilight",
                "in bright pixel dawn", "under clear pixel night sky",
                "in clean pixel morning mist", "during defined pixel storm"
            ])
            
            base_prompt = f"{self.special_tokens['category']}forests{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {forest}, {elements}, {time_of_day}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "everyday":
            item = random.choice(self.everyday_items)
            context = random.choice(self.everyday_contexts)
            detail = random.choice([
                "with pixel-perfect details and clean textures",
                "with distinct pixel materials and sharp edges",
                "with clear pixel shapes and defined features",
                "with precise pixel details and clean lines",
                "with clear pixel form and distinct character"
            ])
            
            base_prompt = f"{self.special_tokens['category']}everyday{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {item}, {context}, {detail}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "objects":
            subcategory = random.choice(list(self.object_types.keys()))
            object_desc = random.choice(self.object_types[subcategory])
            context = random.choice(self.everyday_contexts)
            
            base_prompt = f"{self.special_tokens['category']}objects{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {object_desc}, {context}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "scenes":
            subcategory = random.choice(list(self.scene_types.keys()))
            scene_desc = random.choice(self.scene_types[subcategory])
            time_of_day = random.choice([
                "during pixel-perfect golden hour", "at pixel twilight",
                "in bright pixel daylight", "under pixel starlight",
                "in pixel morning mist", "during pixel sunset"
            ])
            
            base_prompt = f"{self.special_tokens['category']}scenes{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {scene_desc}, {time_of_day}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "creatures":
            subcategory = random.choice(list(self.creature_types.keys()))
            creature_desc = random.choice(self.creature_types[subcategory])
            environment = random.choice(self.backgrounds)
            
            base_prompt = f"{self.special_tokens['category']}creatures{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {creature_desc}, in {environment}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "emotions":
            subcategory = random.choice(list(self.emotion_types.keys()))
            emotion_desc = random.choice(self.emotion_types[subcategory])
            atmosphere = random.choice([
                "with pixel-perfect mood", "with subtle pixel feeling",
                "with strong pixel emotion", "with gentle pixel expression",
                "with powerful pixel impact", "with delicate pixel sentiment"
            ])
            
            base_prompt = f"{self.special_tokens['category']}emotions{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {emotion_desc}, {atmosphere}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "concepts":
            subcategory = random.choice(list(self.concept_types.keys()))
            concept_desc = random.choice(self.concept_types[subcategory])
            interpretation = random.choice([
                "with pixel metaphorical elements",
                "with pixel symbolic representation",
                "with pixel abstract interpretation",
                "with pixel visual storytelling",
                "with pixel conceptual design"
            ])
            
            base_prompt = f"{self.special_tokens['category']}concepts{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {concept_desc}, {interpretation}, {quality_emphasis}, {tech_specs}, with {lighting}"
            
        elif category == "styles":
            subcategory = random.choice(list(self.style_variations.keys()))
            style_desc = random.choice(self.style_variations[subcategory])
            enhancement = random.choice([
                "with pixel genre-specific details",
                "with pixel thematic elements",
                "with pixel style-appropriate effects",
                "with pixel characteristic features",
                "with pixel distinctive atmosphere"
            ])
            
            base_prompt = f"{self.special_tokens['category']}styles{self.special_tokens['style']}{style}{self.special_tokens['detail']}A pixel art {style_desc}, {enhancement}, {quality_emphasis}, {tech_specs}, with {lighting}"
        
        # Add color and pixel detail styles with more emphasis on pixel art techniques
        prompt = f"{base_prompt}{self.special_tokens['color']}{color_style}, {pixel_detail}{self.special_tokens['end']}"
        
        if len(prompt.split()) > self.max_seq_length:
            prompt = " ".join(prompt.split()[:self.max_seq_length])
        
        return prompt, category

    def generate_batch(self, batch_size=10, category="random", start_index=0, used_prompts=set()):
        metadata = []
        prompts_data = []
        
        for i in range(batch_size):
            try:
                prompt, cat = self.generate_prompt(category)
                if prompt in used_prompts:
                    continue
                used_prompts.add(prompt)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"pixel_art_{start_index + i:04d}_{timestamp}.png"
                
                logging.info(f"Generating image {start_index + i + 1}/{batch_size} - Category: {cat} - Batch Progress: {((i+1)/batch_size)*100:.1f}%")
                
                params = self.generation_params[cat]
                seed = random.randint(0, 999999)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                
                # Generate the image
                image = self.pipe(
                    prompt=prompt,
                    height=128,
                    width=128,
                    guidance_scale=params['guidance_scale'],
                    pag_scale=params['pag_scale'],
                    num_inference_steps=params['num_steps'],
                    generator=generator
                )[0]
                
                processed_image = self.pixelate_image(image[0], pixel_size=params['pixel_size'])
                
                # Save the image in PNG format
                processed_image.save(self.images_dir / image_filename)
                
                # Convert to array and store for sprites.npy
                img_array = np.array(processed_image)
                self.all_images.append(img_array)
                
                # Extended metadata
                image_metadata = {
                    "filename": image_filename,
                    "prompt": prompt,
                    "category": cat,
                    "timestamp": timestamp,
                    "seed": seed,
                    "resolution": "128x128",
                    "pixel_size": params['pixel_size'],
                    "image_index": start_index + i,
                    "generation_params": params
                }
                
                metadata.append(image_metadata)
                self.all_metadata.append(image_metadata)
                
                prompts_data.append({
                    "filename": image_filename,
                    "prompt": prompt,
                    "category": cat
                })
                
                logging.info(f"Successfully saved {image_filename}")
                
            except Exception as e:
                logging.error(f"Error generating image: {str(e)}")
                continue
        
        # Save batch metadata
        try:
            metadata_filename = f"metadata_batch_{start_index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(self.metadata_dir / metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            prompts_filename = f"prompts_batch_{start_index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(self.prompts_dir / prompts_filename, 'w') as f:
                json.dump(prompts_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving batch metadata: {str(e)}")

    def save_final_dataset(self):
        """Salva o dataset completo em formato CSV e NPY."""
        logging.info("Salvando dataset final...")
        
        # Check if we have any images to save
        if not self.all_images:
            logging.warning("No images were generated. Creating empty dataset files.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create empty files to avoid errors
            np.save(self.output_dir / f"sprites_{timestamp}.npy", np.array([]))
            import pandas as pd
            pd.DataFrame(columns=['filename', 'category', 'prompt', 'seed', 'pixel_size', 
                                'guidance_scale', 'pag_scale', 'num_steps']).to_csv(
                self.output_dir / f"labels_{timestamp}.csv", index=False)
            return
        
        # Get timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_num = len(list(self.output_dir.glob("sprites_*.npy")))
        
        # Salva sprites.npy with unique name
        sprites_array = np.stack(self.all_images)
        sprites_filename = f"sprites_{timestamp}_batch{batch_num:03d}.npy"
        np.save(self.output_dir / sprites_filename, sprites_array)
        logging.info(f"Sprites salvos em {sprites_filename} com shape {sprites_array.shape}")
        
        # Prepara dados para CSV
        csv_data = []
        for metadata in self.all_metadata:
            csv_row = {
                'filename': metadata['filename'],
                'category': metadata['category'],
                'prompt': metadata['prompt'],
                'seed': metadata['seed'],
                'pixel_size': metadata['pixel_size'],
                'guidance_scale': metadata['generation_params']['guidance_scale'],
                'pag_scale': metadata['generation_params']['pag_scale'],
                'num_steps': metadata['generation_params']['num_steps']
            }
            csv_data.append(csv_row)
        
        # Salva labels.csv with unique name
        import pandas as pd
        df = pd.DataFrame(csv_data)
        labels_filename = f"labels_{timestamp}_batch{batch_num:03d}.csv"
        df.to_csv(self.output_dir / labels_filename, index=False)
        logging.info(f"Labels salvos em {labels_filename} com {len(df)} entradas")

def main():
    generator = PixelArtGenerator()
    total_desired_images = 2000  # Aumentado de 500 para 2000
    batch_size = 25
    
    # Limpa diretórios
    import shutil
    shutil.rmtree(generator.images_dir, ignore_errors=True)
    shutil.rmtree(generator.metadata_dir, ignore_errors=True)
    shutil.rmtree(generator.prompts_dir, ignore_errors=True)
    generator.images_dir.mkdir(parents=True, exist_ok=True)
    generator.metadata_dir.mkdir(parents=True, exist_ok=True)
    generator.prompts_dir.mkdir(parents=True, exist_ok=True)
    generator.existing_images = 0
    
    logging.info(f"Starting generation of {total_desired_images} pixel art images (Characters and Food)")
    logging.info(f"Will generate images in batches of {batch_size}")
    
    # Conjunto para rastrear prompts usados e evitar repetições
    used_prompts = set()
    
    try:
        num_batches = (total_desired_images + batch_size - 1) // batch_size
        for batch_num in range(num_batches):
            try:
                current_batch_size = min(batch_size, total_desired_images - batch_num * batch_size)
                i = batch_num * batch_size
                start_index = i
                
                # Alterna categorias de forma mais equilibrada
                categories = list(generator.style_weights.keys())
                category = categories[batch_num % len(categories)]
                
                logging.info(f"\nStarting batch {batch_num + 1}/{num_batches} ({current_batch_size} images) - Category: {category}")
                generator.generate_batch(
                    batch_size=current_batch_size,
                    category=category,
                    start_index=start_index,
                    used_prompts=used_prompts  # Passa o conjunto de prompts usados
                )
                
            except KeyboardInterrupt:
                logging.info("\n\nProcess interrupted by user!")
                generator.save_final_dataset()  # Salva o dataset mesmo se interrompido
                logging.info("Dataset parcial salvo com sucesso!")
                return
                
    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
    finally:
        # Salva o dataset completo no final
        generator.save_final_dataset()
        logging.info("\nGeneration process completed")
        logging.info(f"Total images generated: {len(list(generator.images_dir.glob('*.png')))}")
        logging.info(f"Unique prompts used: {len(used_prompts)}")

if __name__ == "__main__":
    import signal
    
    def signal_handler(sig, frame):
        logging.info("\n\nReceived interrupt signal (Ctrl+C)")
        logging.info("Please wait while we clean up and save progress...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    main()