# src/extraction/schemas.py
"""
Module: schemas
Functionality: Defines Pydantic models for validating the structure and data types
               of the information extracted by the LLM for the membrane materials project.
               This ensures data consistency and quality before it's saved or used
               in downstream tasks.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator, field_validator


# --- Base Schema Configuration ---
class BaseSchema(BaseModel):
    """Base schema with common configurations for all models in this project."""
    model_config = {
        "extra": "allow",  # Allow extra fields not explicitly defined in the schema
        "validate_assignment": True,
        "str_strip_whitespace": True, # Strip leading/trailing whitespace from strings
    }

# --- Detailed Sub-Schemas for Membrane Materials ---

class MembraneAdditiveSchema(BaseModel):
    name: Optional[str] = Field(default=None, alias="Name", description="Name or type of additive (e.g., PVP, GO, LiCl).")
    concentration_text: Optional[str] = Field(default=None, alias="ConcentrationText", description="Full additive concentration as text, e.g., '0.5 wt.% based on polymer'.")
    role: Optional[str] = Field(default=None, alias="Role", description="Role of the additive, e.g., 'pore former', 'hydrophilicity enhancer', 'nanofiller'.")

class MembranePolymerSchema(BaseSchema):
    name: Optional[str] = Field(default=None, alias="Name", description="Name or abbreviation of the polymer (e.g., 'PES', 'Polyethersulfone').")
    concentration_text: Optional[str] = Field(default=None, alias="ConcentrationText", description="Full polymer concentration as text, e.g., '18 wt.%'.")
    molecular_weight: Optional[str] = Field(default=None, alias="MolecularWeight", description="Molecular weight of the polymer, e.g., '50 kDa'.")

class MembraneSolventSchema(BaseSchema):
    name: Optional[str] = Field(default=None, alias="Name", description="Name or abbreviation of the solvent (e.g., 'NMP', 'DMF', 'DMAc').")
    ratio: Optional[str] = Field(default=None, alias="Ratio", description="Ratio if a solvent mixture is used, e.g., 'NMP/THF (70/30 v/v)'.")

class MembraneDesignSchema(BaseSchema):
    base_polymer: Optional[List[MembranePolymerSchema]] = Field(default_factory=list, alias="BasePolymer", description="List of base polymers used and their concentrations.")
    support_material: Optional[Union[str, List[str]]] = Field(default=None, alias="SupportMaterial", description="Material of the support layer. Can be a single string or a list if multiple are mentioned.")
    active_layer_monomers: Optional[List[str]] = Field(default_factory=list, alias="ActiveLayerMonomers", description="Monomers for the active layer in TFC membranes (e.g., ['MPD', 'TMC']).")
    solvents: Optional[List[MembraneSolventSchema]] = Field(default_factory=list, alias="Solvents", description="Solvents used in the dope solution.")
    additives: Optional[List[MembraneAdditiveSchema]] = Field(default_factory=list, alias="Additives", description="List of additives in the dope solution.")
    overall_composition_notes: Optional[str] = Field(default=None, alias="OverallCompositionNotes", description="General notes on membrane composition or design strategy.")

    # --- 新增的验证器 ---
    @field_validator('base_polymer', mode='before')
    @classmethod
    def wrap_single_polymer_in_list(cls, v: Any) -> Any:
        """
        Catches cases where the LLM provides a single polymer dict instead of a list of dicts.
        If the input 'v' for 'base_polymer' is a dictionary, this validator wraps it in a list
        before the main validation for List[MembranePolymerSchema] occurs.
        """
        if isinstance(v, dict):
            # 如果输入是一个字典，将它包装成一个列表
            return [v]
        # 如果输入已经是列表、None或其它类型，则保持原样，让后续的标准验证器处理
        return v
    # --- 验证器结束 ---
class MembraneCastingCoatingSchema(BaseSchema):
    solution_temperature: Optional[str] = Field(default=None, alias="CastingSolutionTemperature", description="Temperature of the casting solution, e.g., '25°C'.")
    casting_thickness: Optional[str] = Field(default=None, alias="CastingThickness", description="Thickness of the cast film before phase inversion, e.g., '200 µm'.")
    casting_speed: Optional[str] = Field(default=None, alias="CastingSpeed", description="Speed of casting or coating.")
    environment_humidity: Optional[str] = Field(default=None, alias="EnvironmentHumidity", description="Relative humidity during casting/evaporation, e.g., '50% RH'.")
    evaporation_time: Optional[str] = Field(default=None, alias="EvaporationTime", description="Solvent evaporation time before immersion, e.g., '30 s'.")

class MembraneCoagulationBathSchema(BaseSchema):
    composition: Optional[str] = Field(default=None, alias="CoagulationBathComposition", description="Composition of the coagulation bath, e.g., 'Pure water', 'Water/IPA mixture'.")
    temperature: Optional[str] = Field(default=None, alias="CoagulationBathTemperature", description="Temperature of the coagulation bath, e.g., '25°C'.")
    immersion_time: Optional[str] = Field(default=None, alias="ImmersionTime", description="Duration of immersion in coagulation bath, e.g., '10 min'.")

class MembranePostTreatmentSchema(BaseSchema):
    treatment_type: Optional[str] = Field(default=None, alias="TreatmentType", description="Type of post-treatment, e.g., 'Annealing', 'Crosslinking', 'Washing'.")
    temperature: Optional[str] = Field(default=None, alias="Temperature", description="Temperature for thermal treatments, e.g., '80°C'.")
    duration: Optional[str] = Field(default=None, alias="Duration", description="Duration of the post-treatment, e.g., '2 hours'.")
    atmosphere_or_medium: Optional[str] = Field(default=None, alias="AtmosphereOrMedium", description="Atmosphere or medium for post-treatment, e.g., 'Glycerol solution'.")
    reagents: Optional[str] = Field(default=None, alias="Reagents", description="Chemicals used in post-treatment, e.g., 'Glutaraldehyde'.")

class MembraneFabricationSchema(BaseSchema):
    primary_method: Optional[str] = Field(default=None, alias="FabricationMethod", description="Main fabrication technique, e.g., 'NIPS', 'Interfacial Polymerization'.")
    casting_coating_details: Optional[MembraneCastingCoatingSchema] = Field(default_factory=MembraneCastingCoatingSchema, alias="CastingCoatingDetails")
    coagulation_bath_details: Optional[MembraneCoagulationBathSchema] = Field(default_factory=MembraneCoagulationBathSchema, alias="CoagulationBathDetails")
    interfacial_polymerization_details: Optional[str] = Field(default=None, alias="InterfacialPolymerizationDetails", description="Specifics of IP process if applicable.")
    electrospinning_details: Optional[str] = Field(default=None, alias="ElectrospinningDetails", description="Specifics of electrospinning process.")
    surface_modification_method: Optional[str] = Field(default=None, alias="SurfaceModificationMethod", description="Method used for surface modification, if any.")
    post_treatment_details: Optional[List[MembranePostTreatmentSchema]] = Field(default_factory=list, alias="PostTreatmentDetails", description="List of post-treatment steps.")
    final_film_thickness_text: Optional[str] = Field(default=None, alias="FinalFilmThicknessText", description="Thickness of the final membrane, e.g., '120 µm'.")
    fabrication_notes: Optional[str] = Field(default=None, alias="FabricationNotes", description="Other general notes on fabrication.")

class MembraneStructuralPhysicalPropertiesSchema(BaseSchema):
    type: Optional[str] = Field(default=None, alias="Type", description="Membrane classification, e.g., 'UF', 'NF', 'RO'.")
    morphology: Optional[str] = Field(default=None, alias="Morphology", description="e.g., 'Asymmetric', 'Sponge-like pores'.")
    pore_size_text: Optional[str] = Field(default=None, alias="PoreSizeText", description="Reported pore size, e.g., '20 nm'.")
    mwco_text: Optional[str] = Field(default=None, alias="MWCOText", description="Molecular Weight Cut-Off, e.g., '10 kDa'.")
    porosity_text: Optional[str] = Field(default=None, alias="Porosity", description="e.g., '75%'.")
    surface_roughness_rms: Optional[str] = Field(default=None, alias="SurfaceRoughnessRMS", description="RMS surface roughness, e.g., '5 nm'.")
    contact_angle_text: Optional[str] = Field(default=None, alias="ContactAngleText", description="Water contact angle, e.g., '65°'.")
    zeta_potential_text: Optional[str] = Field(default=None, alias="ZetaPotentialText", description="e.g., '-20 mV at pH 7'.")
    swelling_degree_text: Optional[str] = Field(default=None, alias="SwellingDegreeText", description="e.g., '15% in water'.")

class MembraneLiquidTransportPropertiesSchema(BaseSchema):
    pure_water_flux: Optional[str] = Field(default=None, alias="WaterFlux", description="e.g., '150 L m⁻² h⁻¹ at 2 bar'.")
    pure_water_permeability: Optional[str] = Field(default=None, alias="WaterPermeability", description="e.g., '75 L m⁻² h⁻¹ bar⁻¹'.")
    solvent_flux: Optional[str] = Field(default=None, alias="SolventFlux", description="e.g., 'Ethanol flux: 30 LMH at 5 bar'.")
    rejections: Optional[Dict[str, str]] = Field(default_factory=dict, alias="Rejections", description="Dictionary of solute rejections, e.g., {'NaCl': '98%', 'BSA': '>99%'}.")
    fouling_resistance_notes: Optional[str] = Field(default=None, alias="FoulingResistanceNotes", description="Description of antifouling tests.")
    flux_recovery_ratio: Optional[str] = Field(default=None, alias="FluxRecoveryRatio", description="FRR value, e.g., '92%'.")

class MembraneGasTransportPropertiesSchema(BaseSchema):
    permeances: Optional[Dict[str, str]] = Field(default_factory=dict, alias="Permeances", description="Dictionary of gas permeances, e.g., {'CO2': '1000 GPU'}.")
    permeabilities: Optional[Dict[str, str]] = Field(default_factory=dict, alias="Permeabilities", description="Dictionary of gas permeabilities, e.g., {'CO2': '100 Barrer'}.")
    selectivities: Optional[Dict[str, str]] = Field(default_factory=dict, alias="Selectivities", description="Dictionary of gas pair selectivities, e.g., {'CO2/N2': '50'}.")

class MembraneMechanicalPropertiesSchema(BaseSchema):
    tensile_strength: Optional[str] = Field(default=None, alias="TensileStrength", description="e.g., '5 MPa'.")
    youngs_modulus: Optional[str] = Field(default=None, alias="YoungsModulus", description="e.g., '1.2 GPa'.")
    elongation_at_break: Optional[str] = Field(default=None, alias="ElongationAtBreak", description="e.g., '150%'.")

class MembraneStabilitySchema(BaseSchema):
    thermal_stability: Optional[str] = Field(default=None, alias="ThermalStability", description="e.g., 'Stable up to 150°C'.")
    chemical_stability: Optional[str] = Field(default=None, alias="ChemicalStability", description="e.g., 'Stable in pH range 2-12'.")
    operational_stability: Optional[str] = Field(default=None, alias="OperationalStability", description="e.g., 'Stable flux over 100 hours'.")

class MembranePerformanceSchema(BaseSchema):
    structural_physical_properties: Optional[MembraneStructuralPhysicalPropertiesSchema] = Field(default_factory=MembraneStructuralPhysicalPropertiesSchema, alias="StructuralPhysicalProperties")
    liquid_transport_properties: Optional[MembraneLiquidTransportPropertiesSchema] = Field(default_factory=MembraneLiquidTransportPropertiesSchema, alias="LiquidTransportProperties")
    gas_transport_properties: Optional[MembraneGasTransportPropertiesSchema] = Field(default_factory=MembraneGasTransportPropertiesSchema, alias="GasTransportProperties")
    mechanical_properties: Optional[MembraneMechanicalPropertiesSchema] = Field(default_factory=MembraneMechanicalPropertiesSchema, alias="MechanicalProperties")
    stability_properties: Optional[MembraneStabilitySchema] = Field(default_factory=MembraneStabilitySchema, alias="StabilityProperties")
    general_performance_summary: Optional[str] = Field(default=None, alias="GeneralPerformanceSummary", description="A general summary if specific values are not broken down.")

class MembraneApplicationFeedSchema(BaseSchema):
    composition: Optional[str] = Field(default=None, alias="FeedSolution", description="Detailed composition of the feed solution/gas mixture.")
    concentration: Optional[str] = Field(default=None, alias="Concentration", description="Concentration of key components in the feed.")
    temperature: Optional[str] = Field(default=None, alias="Temperature", description="Feed temperature.")
    ph: Optional[str] = Field(default=None, alias="pH", description="Feed pH.")

class MembraneApplicationOperatingConditionsSchema(BaseSchema):
    pressure: Optional[str] = Field(default=None, alias="OperatingPressure", description="Operating pressure or transmembrane pressure (TMP).")
    temperature: Optional[str] = Field(default=None, alias="OperatingTemperature", description="Operating temperature.")
    flow_rate_or_velocity: Optional[str] = Field(default=None, alias="FlowRateOrVelocity", description="Cross-flow velocity or feed flow rate.")
    duration: Optional[str] = Field(default=None, alias="Duration", description="Duration of the application test.")

class MembraneApplicationSchema(BaseSchema):
    application_area: Optional[str] = Field(default=None, alias="ApplicationArea", description="Specific field of application, e.g., 'Brackish water desalination', 'CO2 capture'.")
    feed_details: Optional[MembraneApplicationFeedSchema] = Field(default_factory=MembraneApplicationFeedSchema, alias="FeedDetails")
    operating_conditions: Optional[MembraneApplicationOperatingConditionsSchema] = Field(default_factory=MembraneApplicationOperatingConditionsSchema, alias="OperatingConditions")
    achieved_performance_in_application: Optional[str] = Field(default=None, alias="AchievedPerformanceInApplication", description="Key results from the application testing, e.g., '99% NaCl rejection from 2000 ppm feed'.")

class MembraneSpecificDetailsSchema(BaseSchema): # This is what goes into "Details" for a Membrane entry
    Design: Optional[MembraneDesignSchema] = Field(default_factory=MembraneDesignSchema)
    Fabrication: Optional[MembraneFabricationSchema] = Field(default_factory=MembraneFabricationSchema)
    Performance: Optional[MembranePerformanceSchema] = Field(default_factory=MembranePerformanceSchema)
    Application: Optional[MembraneApplicationSchema] = Field(default_factory=MembraneApplicationSchema)

    @model_validator(mode='before')
    @classmethod
    def ensure_detail_sections_are_dicts(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for section_name in ["Design", "Fabrication", "Performance", "Application"]:
                if data.get(section_name) is None or not isinstance(data.get(section_name), dict):
                    data[section_name] = {} # Ensure section exists as a dict for sub-schema validation
        return data

# --- Main Schemas for LLM Output and Individual Entries (Generic part) ---
class ExtractedMaterialEntrySchema(BaseSchema):
    """Schema for a single extracted material entry (can be TCO or Membrane)."""
    MaterialName: str = Field(..., min_length=1, description="Descriptive name of the extracted material/sample.")
    Details: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific details. Will be validated by a specific domain schema in the parser.")

class LLMOutputSchema(BaseModel):
    """
    Defines the expected root structure of the JSON returned by the LLM,
    after initial key standardization (e.g., "output" to "Output").
    """
    model_config = { "str_strip_whitespace": True }

    Output: List[ExtractedMaterialEntrySchema] = Field(default_factory=list, description="List of extracted material entries from the paper.")

    @model_validator(mode='before')
    @classmethod
    def ensure_output_is_list_and_not_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            output_val = data.get('Output')
            if output_val is None:
                data['Output'] = []
            elif not isinstance(output_val, list):
                data['Output'] = [output_val]
        elif data is None:
            return {"Output": []}
        return data