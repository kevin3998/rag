# src/config/domain_specific_configs.py
"""
Module: domain_specific_configs
Functionality: Defines and manages domain-specific configurations.
               This includes the DomainConfig class and instances for different
               domains (e.g., "membrane", "in2o3_tco"), specifying keywords,
               blacklists, and field mappings used in the extraction process.
"""
import regex as re
from typing import Dict, Any


class DomainConfig:
    """Stores domain-specific configurations for the extraction pipeline."""

    def __init__(self, domain_name: str, keyword_groups: dict, blacklist: dict, field_mapping: dict):
        self.domain = domain_name
        self.keyword_groups = keyword_groups
        self.blacklist = blacklist
        self.field_mapping = field_mapping

        self.patterns = {
            cat: re.compile(f"{group.get('en', '')}|{group.get('zh', '')}", re.IGNORECASE)
            for cat, group in self.keyword_groups.items() if isinstance(group, dict)
        }
        materials_group = self.keyword_groups.get('materials', {})
        self.material_pattern = re.compile(
            f"{materials_group.get('en', '')}|{materials_group.get('zh', '')}",
            re.IGNORECASE | re.UNICODE
        ) if materials_group and (materials_group.get('en') or materials_group.get('zh')) else None

        blacklist_en = self.blacklist.get('en', '')
        blacklist_zh = self.blacklist.get('zh', '')
        self.blacklist_pattern = re.compile(
            f"{blacklist_en}|{blacklist_zh}",
            re.IGNORECASE
        ) if blacklist_en or blacklist_zh else None

    def is_domain_related(self, text: str) -> bool:
        """Checks if a text is relevant to the domain based on keywords and blacklist."""
        if not self.material_pattern:
            return True  # If no specific material keywords, assume it's relevant by default.

        is_related = self.material_pattern.search(text) is not None
        if self.blacklist_pattern:
            is_blacklisted = self.blacklist_pattern.search(text) is not None
            return is_related and not is_blacklisted
        return is_related

    def count_keywords(self, text: str) -> int:
        """Counts the total number of keyword occurrences in a text."""
        return sum(len(pattern.findall(text)) for pattern in self.patterns.values())


# --- Config for Membrane Materials Domain ---

# Keywords to identify relevant documents and concepts
MEMBRANE_KEYWORDS = {
    "materials": {
        "en": r"(?i)\b(PSF|Polysulfone|PES|Polyethersulfone|PVDF|Polyvinylidene\s*fluoride|PTFE|Polytetrafluoroethylene|Polyamide|PA|TFC|Thin\s*Film\s*Composite|TFN|Thin\s*Film\s*Nanocomposite|PAN|Polyacrylonitrile|Ceramic\s*Membrane|Al2O3|TiO2|ZrO2|Zeolite|Nafion|PIMs?|Polymer\s*of\s*Intrinsic\s*Microporosity|PEI|Polyetherimide|Cellulose\s*Acetate|CA)\b",
        "zh": r"(聚砜|PSF|聚醚砜|PES|聚偏氟乙烯|PVDF|聚四氟乙烯|PTFE|聚酰胺|PA|薄膜复合|TFC|陶瓷膜|聚丙烯腈|PAN|醋酸纤维素)"
    },
    "fabrication_methods": {
        "en": r"(?i)\b(phase\s*inversion|NIPS|interfacial\s*polymerization|IP|electrospinning|sol-gel|dip\s*coating|spin\s*coating|surface\s*modification|grafting|cross-?linking|annealing|heat\s*treatment)\b",
        "zh": r"(相转化|非溶剂致相分离|界面聚合|静电纺丝|溶胶凝胶|涂覆|表面改性|接枝|交联|退火)"
    },
    "properties_performance": {
        "en": r"(?i)\b(permeance|permeability|flux|water\s*flux|rejection|retention|selectivity|separation\s*factor|MWCO|molecular\s*weight\s*cut-?off|pore\s*size|porosity|contact\s*angle|hydrophilicity|zeta\s*potential|mechanical\s*strength|tensile\s*strength|antifouling|FRR|flux\s*recovery\s*ratio|gas\s*separation|O2/N2|CO2/N2|H2/CO2)\b",
        "zh": r"(渗透性|通量|水通量|截留率|选择性|截留分子量|孔径|孔隙率|接触角|亲水性|机械强度|抗污染|气体分离)"
    },
    "applications": {
        "en": r"(?i)\b(seawater\s*for\s*hydrogen|hydrogen\s*production|industrial\s*wastewater|dyeing\s*wastewater|textile\s*wastewater|air\s*separation|water\s*treatment|desalination|reverse\s*osmosis|RO|nanofiltration|NF|ultrafiltration|UF|microfiltration|MF|gas\s*separation|CO2\s*capture|hydrogen\s*purification|pervaporation|PV|membrane\s*bioreactor|MBR)\b",
        "zh": r"(海水制氢|工业废水|印染废水|空气分离|水处理|海水淡化|反渗透|纳滤|超滤|微滤|气体分离)"
    },
    "additives_modifiers": {
        "en": r"(?i)\b(additive|nanoparticle|PVP|polyvinylpyrrolidone|PEG|polyethylene\s*glycol|LiCl|graphene\s*oxide|GO|CNT|carbon\s*nanotube|TiO2|ZnO|Ag|MOF|ZIF)\b",
        "zh": r"(添加剂|纳米粒子|PVP|PEG|氯化锂|氧化石墨烯|碳纳米管)"
    }
}

# Terms to exclude documents that are likely out of scope
MEMBRANE_BLACKLIST = {
    "en": r"(?i)\b(cell\s*membrane|biological\s*membrane|lipid\s*bilayer|ion\s*channel|packaging\s*film\s*not\s*for\s*separation|fuel\s*cell\s*electrode\s*not\s*the\s*membrane)\b",
    "zh": r"(细胞膜|生物膜|脂质双分子层|离子通道)"
}

# Mapping from potential LLM output keys to your standardized internal keys (Pydantic aliases)
MEMBRANE_FIELD_MAPPING = {
    "en": {
        # Top-level categories for Details object
        "Design": ["Design", "Membrane Design", "Material Composition"],
        "Fabrication": ["Fabrication", "Membrane Preparation", "Synthesis"],
        "Performance": ["Performance", "Membrane Properties", "Separation Performance", "Characterization"],
        "Application": ["Application", "Application Field", "Usage Scenarios"],

        # Design sub-fields
        "BasePolymer": ["BasePolymer", "Base Polymer", "Matrix Polymer", "Main Polymer"],
        "PolymerConcentration": ["PolymerConcentration", "Polymer Concentration", "Dope Solution Concentration",
                                 "Casting Solution Concentration"],
        "Solvent": ["Solvent", "Solvent System"],
        "Additive": ["Additive", "Modifier", "Filler", "Nanoparticle", "Additives"],  # Plural and singular
        "AdditiveConcentration": ["AdditiveConcentration", "Additive Concentration", "Filler Loading",
                                  "Nanoparticle Loading"],
        "SupportMaterial": ["SupportMaterial", "Support Material", "Support Layer"],
        "ActiveLayerMonomers": ["ActiveLayerMonomers", "Monomers", "Aqueous Phase Monomer", "Organic Phase Monomer"],

        # Fabrication sub-fields
        "FabricationMethod": ["FabricationMethod", "Preparation Method", "Technique"],
        "FilmThicknessText": ["FilmThicknessText", "Film Thickness", "Membrane Thickness"],
        "CastingSolutionTemperature": ["CastingSolutionTemperature", "Casting Solution Temperature",
                                       "Dope Temperature"],
        "EvaporationTime": ["EvaporationTime", "Evaporation Time", "Exposure Time"],
        "CoagulationBathComposition": ["CoagulationBathComposition", "Coagulation Bath",
                                       "Non-solvent Bath Composition"],
        "CoagulationBathTemperature": ["CoagulationBathTemperature", "Coagulation Bath Temperature"],
        "AnnealingTemperature": ["AnnealingTemperature", "Annealing Temperature", "Heat Treatment Temperature"],
        "AnnealingDuration": ["AnnealingDuration", "Annealing Duration", "Heat Treatment Time"],
        "CrosslinkingAgent": ["CrosslinkingAgent", "Crosslinking Agent", "Crosslinker"],

        # Performance sub-fields
        "WaterFlux": ["WaterFlux", "Water Flux", "Pure Water Flux", "PWF"],
        "WaterPermeability": ["WaterPermeability", "Water Permeability", "Permeance (Water)", "Pure Water Permeance"],
        "SaltRejection": ["SaltRejection", "Salt Rejection", "NaCl Rejection", "MgSO4 Rejection"],
        "DyeRejection": ["DyeRejection", "Dye Rejection", "Congo Red Rejection", "Methylene Blue Rejection"],
        "GasPermeance": ["GasPermeance", "Permeance"],  # Could be a dict if multiple gases
        "GasPermeability": ["GasPermeability", "Permeability"],  # Could be a dict
        "Selectivity": ["Selectivity", "Ideal Selectivity", "Separation Factor"],  # Could be a dict
        "PoreSizeText": ["PoreSizeText", "Pore Size", "Average Pore Size", "Mean Pore Size"],
        "MWCOText": ["MWCOText", "MWCO", "Molecular Weight Cut Off", "Molecular Weight Cutoff"],
        "Porosity": ["Porosity"],
        "ContactAngleText": ["ContactAngleText", "Contact Angle", "Water Contact Angle", "WCA"],
        "TensileStrength": ["TensileStrength", "Tensile Strength", "Mechanical Strength"],
        "FluxRecoveryRatio": ["FluxRecoveryRatio", "Flux Recovery Ratio", "FRR"],

        # Application sub-fields
        "ApplicationArea": ["ApplicationArea", "Application Scenario", "Target Application", "Field of Use"],
        "FeedSolution": ["FeedSolution", "Feed Composition", "Source Water"],
        "OperatingPressure": ["OperatingPressure", "Operating Pressure", "Applied Pressure", "Transmembrane Pressure",
                              "TMP"],
        "OperatingTemperature": ["OperatingTemperature", "Operating Temperature", "Test Temperature"],
        "AchievedPerformanceInApplication": ["AchievedPerformanceInApplication", "Performance in Application",
                                             "Application Performance"],
    },
    "zh": {
        "Design": ["设计", "膜设计", "材料组成"],
        "Fabrication": ["制备", "膜制备方法", "合成工艺"],
        "Performance": ["性能", "膜性能", "分离性能"],
        "Application": ["应用", "应用领域"],
        "WaterFlux": ["水通量", "纯水通量"],
        "SaltRejection": ["脱盐率", "盐截留率"],
        # ... more Chinese mappings as needed
    }
}

MEMBRANE_CONFIG = DomainConfig(
    domain_name="membrane",
    keyword_groups=MEMBRANE_KEYWORDS,
    blacklist=MEMBRANE_BLACKLIST,
    field_mapping=MEMBRANE_FIELD_MAPPING
)

# --- Example for In2O3 TCO Domain (Kept to show extensibility) ---
# For a full implementation, you would define IN2O3_TCO_KEYWORDS and IN2O3_TCO_FIELD_MAPPING here
# as we did in the previous step. For brevity, it's shortened here.
IN2O3_TCO_CONFIG = DomainConfig(
    domain_name="in2o3_tco",
    keyword_groups={"materials": {"en": r"\b(In2O3|ITO)\b"}},
    blacklist={},
    field_mapping={"en": {"Design": ["Design"], "Fabrication": ["Fabrication"], "Performance": ["Performance"],
                          "Application": ["Application"]}}
)

# --- Global Dictionary of Domain Configurations ---
DOMAIN_CONFIGURATIONS: Dict[str, DomainConfig] = {
    "membrane": MEMBRANE_CONFIG,
    "in2o3_tco": IN2O3_TCO_CONFIG,  # This allows you to run for TCOs by changing the --domain flag
}


def get_domain_config(domain_name: str) -> DomainConfig:
    """Retrieves the configuration object for a given domain name."""
    config = DOMAIN_CONFIGURATIONS.get(domain_name)
    if not config:
        raise ValueError(f"DomainConfig not found for domain: '{domain_name}'")
    return config 