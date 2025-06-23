"""
Module: prompt_templates
Functionality: Stores and manages prompt templates used for LLM interactions.
               Provides functions to load these templates into a PromptManager
               instance, allowing for domain-specific and language-specific prompts.
"""
from extractor.utils.general_utils import PromptManager # Corrected import path

# --- Define prompt templates here ---

MEMBRANE_PROMPT_EN = """You are an expert AI assistant specializing in membrane science and technology. You are tasked with extracting detailed, structured information from the provided text of a research paper.

Please extract information for **each distinct membrane material or sample with unique properties/fabrication conditions**. If multiple membranes are described (e.g., a base PSF membrane and a modified PSF-GO nanocomposite membrane), generate a separate entry for each.

For each distinct membrane, extract and return the following categories of information. Omit any category or sub-field if the information is not present in the text.

1.  **Design (Material Composition & Structure)**:
    * `BasePolymer`: The primary polymer(s) used (e.g., "PES", "PVDF"). Must be a list of objects, even if there's only one.
    * `PolymerConcentration`: The concentration of the polymer in the dope solution (e.g., "18 wt.%").
    * `Solvent`: The solvent(s) used (e.g., "NMP", "DMF/Acetone mixture").
    * `Additive`: Name of additives, fillers, or nanoparticles (e.g., "PVP K30", "Graphene Oxide (GO)", "LiCl").
    * `AdditiveConcentration`: Concentration of the additive (e.g., "5 wt.% relative to polymer", "0.1 g/L").
    * `SupportMaterial`: Material of the support layer for composite membranes (e.g., "Non-woven PET").
    * `ActiveLayerMonomers`: For TFC membranes, list the monomers (e.g., ["MPD", "TMC"]).

2.  **Fabrication (Membrane Preparation & Treatment)**:
    * `FabricationMethod`: The main technique (e.g., "Non-solvent Induced Phase Separation (NIPS)", "Interfacial Polymerization", "Electrospinning").
    * `FilmThicknessText`: Thickness of the final membrane (e.g., "150 µm").
    * `CastingSolutionTemperature`: Temperature of the casting solution (e.g., "25°C").
    * `EvaporationTime`: Solvent evaporation time before coagulation (e.g., "30 s").
    * `CoagulationBathComposition`: Composition of the non-solvent bath (e.g., "Pure water", "Ethanol/water mixture").
    * `CoagulationBathTemperature`: Temperature of the coagulation bath (e.g., "40°C").
    * `AnnealingTemperature`: Post-treatment annealing temperature (e.g., "80°C").
    * `AnnealingDuration`: Post-treatment annealing duration (e.g., "2 hours").
    * `CrosslinkingAgent`: Chemical used for crosslinking if any (e.g., "Glutaraldehyde").

3.  **Performance (Properties & Characterization)**: This section should be a nested object with the following sub-categories:
    * **`StructuralPhysicalProperties`**:
        * `Type`: Membrane classification (e.g., "UF", "NF", "RO", "Gas Separation").
        * `Morphology`: e.g., "Asymmetric with finger-like pores", "Dense selective layer".
        * `PoreSizeText`: Reported pore size (e.g., "20 nm", "0.1 µm").
        * `MWCOText`: Molecular Weight Cut-Off (e.g., "10 kDa", "500 Da").
        * `Porosity`: e.g., "78%".
        * `ContactAngleText`: Water contact angle (e.g., "65°").
        * `SurfaceRoughnessRMS`: RMS roughness (e.g., "5.2 nm").
    * **`LiquidTransportProperties`**:
        * `WaterFlux`: e.g., "150 L m⁻² h⁻¹ (at 2 bar)".
        * `WaterPermeability`: e.g., "75 L m⁻² h⁻¹ bar⁻¹".
        * `Rejections`: A dictionary of solute rejections (e.g., {{"NaCl": "98%", "Congo Red": ">99%"}}).
        * `FluxRecoveryRatio`: FRR value for fouling tests (e.g., "95% after 3 cycles with BSA").
    * **`GasTransportProperties`**:
        * `Permeances`: A dictionary of gas permeances (e.g., {{"CO2": "1000 GPU", "N2": "25 GPU"}}).
        * `Selectivities`: A dictionary of ideal gas selectivities (e.g., {{"CO2/N2": "40"}}).
    * **`MechanicalProperties`**:
        * `TensileStrength`: e.g., "4.5 MPa".
        * `ElongationAtBreak`: e.g., "80%".

4.  **Application (Usage & Testing Conditions)**:
    * `ApplicationScenario`: The target application (e.g., "Industrial wastewater treatment", "CO2 capture from flue gas", "Brackish water desalination").
    * `FeedSolution`: Composition of the feed used in testing (e.g., "2000 ppm NaCl solution", "Simulated textile wastewater with 100 ppm Reactive Black 5 dye").
    * `OperatingPressure`: e.g., "2 bar", "15 bar".
    * `OperatingTemperature`: e.g., "25°C".
    * `AchievedPerformanceInApplication`: A summary of key results from the application test.

Strictly implement the following requirements:
1.  Return a strict JSON object. The root of this object MUST contain an "output" key, and its value MUST be a list. Each item in the "output" list corresponds to one membrane material/sample.
    Example Structure:
    {{
      "output": [
        {{
          "MaterialName": "PVDF-GO Nanocomposite Membrane (0.5 wt% GO)",
          "Details": {{
            "Design": {{
              "BasePolymer": [{{ "Name": "PVDF", "ConcentrationText": "18 wt.%" }}],
              "Solvents": [{{ "Name": "DMF" }}],
              "Additives": [{{ "Name": "Graphene Oxide (GO)", "ConcentrationText": "0.5 wt.% relative to PVDF" }}]
            }},
            "Fabrication": {{
              "FabricationMethod": "Non-solvent Induced Phase Separation (NIPS)",
              "CastingSolutionTemperature": "30°C",
              "CoagulationBathComposition": "Water",
              "CoagulationBathTemperature": "25°C",
              "FilmThicknessText": "150 µm"
            }},
            "Performance": {{
              "StructuralPhysicalProperties": {{
                "Type": "UF",
                "Porosity": "81.3%",
                "ContactAngleText": "72.5°"
              }},
              "LiquidTransportProperties": {{
                "WaterFlux": "250 L m⁻² h⁻¹ (at 1 bar)",
                "Rejections": {{
                  "Congo Red": ">99%",
                  "BSA": "98.5%"
                }}
              }},
              "MechanicalProperties": {{
                "TensileStrength": "5.2 MPa"
              }},
              "GasTransportProperties": {{}}
            }},
            "Application": {{
              "ApplicationScenario": "Dyeing wastewater treatment",
              "FeedSolution": "50 mg/L Congo Red solution",
              "OperatingPressure": "1 bar"
            }}
          }}
        }}
      ]
    }}
Only return the JSON. Do not include any extra explanations or markdown backticks.
2.  Inside each JSON object within the "output" list, two keys are expected: "MaterialName" and "Details". The **"MaterialName" key is MANDATORY AND MUST NOT BE OMITTED**. The value for "MaterialName" must be a descriptive name for the specific sample being described in that entry (e.g., "PVDF-GO Nanocomposite Membrane (0.5 wt% GO)").
3.  Extract data that corresponds ONLY to a specific, unique material or sample's synthesis and characterization as described in the paper. If the text contains general review tables listing typical properties for a class of materials, **DO NOT** use that data to populate the fields for a specific experimental sample mentioned elsewhere.
4.  All performance metrics MUST be nested inside their respective sub-objects (`ElectricalProperties`, `OpticalProperties`, `StructuralProperties`, `OtherPerformanceMetrics`, `LiquidTransportProperties`, `GasTransportProperties`, `MechanicalProperties`) within the main `Performance` object. Do not place fields like `Resistivity` or `WaterFlux` directly under `Performance`.
5.  If a category, sub-object, or specific field (other than the mandatory "MaterialName") is not mentioned in the text for a specific material, omit that key entirely from the JSON output. Do not use `null`, empty strings, or placeholders like "N/A" for missing information.

Input Text:
{text}
"""
# Note: The prompt's example JSON should be as close as possible to what you desire.
      # The {text} placeholder will be filled by PromptManager.

# You can add more prompts for other domains or languages here

def load_prompts(prompt_manager: PromptManager):
    prompt_manager.add_prompt(domain="membrane", language="en", template=MEMBRANE_PROMPT_EN)
    # prompt_manager.add_prompt(domain="another_domain", language="en", template=ANOTHER_PROMPT_EN)
    # ... add other prompts
    return prompt_manager