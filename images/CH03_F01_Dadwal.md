# Figure 3.1: Self-supervised learning taxonomy

**Status:** editable vector figure included as `CH03_F01_Dadwal.svg`.

This updated objective-centred map replaces the planned family-comparison Table 3.1. It uses the user-supplied 2024 diagram as a visual reference, but groups representative methods by their principal training signal and adds latent-prediction methods publicized in 2025. The taxonomy is explanatory, not exhaustive or mutually exclusive: CPC is both temporal and contrastive, and self-distillation and redundancy reduction can cross family boundaries.

**Families shown**

- Pretext prediction: rotation, jigsaw, and colorization.
- Contrastive identification: SimCLR/NT-Xent, MoCo, and CPC.
- Non-contrastive joint embedding: SimSiam, BYOL/DINO, Barlow Twins, and VICReg.
- Masked modeling and prediction: MAE/SimMIM, BEiT/iBOT, and I-JEPA/V-JEPA 2.
- Cross-modal alignment: CLIP, ImageBind, and video-text objectives.

**Sources:** Gui et al. (2024); Uelwer et al. (2025); Assran et al. (2025). The exact bibliographic entries are in `manuscript/CH03_Dadwal_World_Models_in_Action.adoc`.

**Acceptance checks:** method names are spelled correctly; labels remain legible at book width; arrows and branches remain clear in grayscale; no category implies that families are mutually exclusive.