# PRO-GO: Reference-Guided Protein Sequence Generation Demo

> **⚠️ IMPORTANT NOTICE: This demo is not for public sharing. This repository contains proprietary research and should only be used for internal demonstration purposes.**

This repository contains a Streamlit demo showcasing the PRO-GO paper: "Reference-Guided Protein Sequence Generation using Gene Ontology Terms" by Darren Tan et al.

## Overview

PRO-GO presents a novel framework for controllable protein sequence generation that:
- Uses reference sequences to guide generation
- Specifies target characteristics through Gene Ontology (GO) terms
- Enables flexible generation without model retraining
- Achieves high accuracy in structural similarity

## Files

- `streamlit_demo.py` - Interactive Streamlit demo with ESMFold-style visualization
- `IEEE_ACCESS___PROGO_ver8.pdf` - Original research paper
- `go_term_sets_usecase.csv` - Predefined GO term sets for common therapeutic targets
- `requirements.txt` - Python dependencies including BioPython for PDB analysis

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Black Box in 3D Visualization
If you see a black box instead of the protein structure:
- Ensure you have an active internet connection (structures are loaded from rcsb.org)
- Try refreshing the page
- Check browser console for errors (F12)
- Use Chrome or Edge for best WebGL support

### pLDDT Colors Not Showing (Grey Structure)
If the structure appears grey instead of colored by confidence:
- Make sure you've selected "AlphaFold (with pLDDT scores)" as the structure source
- Ensure "Color by pLDDT/B-factor" checkbox is checked
- Note: Experimental PDB structures don't contain pLDDT scores - only AlphaFold structures do
- Technical note: py3Dmol uses property name "b" (not "bfactor") for B-factor coloring

## Running the Demo

```bash
streamlit run streamlit_demo.py
```

The demo will open in your web browser at `http://localhost:8501`

## Demo Features

The interactive demo includes:

1. **Overview** - Paper abstract and key contributions
2. **Motivation** - Current challenges in protein design and PRO-GO's solution
3. **Methodology** - Framework overview and method steps
4. **Experiments** - Models tested and evaluation metrics
5. **Results** - Performance comparisons and key findings
6. **Key Insights** - Important takeaways from the research
7. **Interactive Demo** - Uses REAL protein data from paper evaluation with:
          - **Predefined therapeutic target use cases** - Choose from curated GO term sets for:
            - Kinase targets (cancer therapy)
            - GPCR receptors (hormone/nerve signaling)
            - Ion channels/transporters (drug flow control)
            - Transcription factors (gene regulation)
            - Immune response proteins (cytokines/receptors)
          - **Complete references** for each use case including:
            - Direct links to GO term definitions on AmiGO
            - DOI links to peer-reviewed scientific papers
            - GO aspects (Molecular Function, Biological Process, Cellular Component)
   - Manual GO term selection for custom targets
   - **Interactive reference sequence retrieval** - Click to search UniRef50 database (~5 seconds)
   - UniRef50 reference sequences with verified GO term annotations clearly displayed
          - Generated sequences designed for target GO terms (confidence scores shown after structure prediction)
          - **Realistic progress animations** for reference retrieval (~5 seconds), sequence generation (~5 seconds), and structure prediction (~5 seconds)
          - Adjustable number of sequences to generate (1-5)
          - Interactive structure prediction workflow - select a sequence then click to predict structure
          - TM-scores revealed after ESMFold prediction (all above 0.8 reflecting GO match accuracy)
          - **Side-by-side structure comparison** - PRO-GO predicted vs ground truth proteins with same GO terms
          - **ESMFold-style 3D protein visualization** using py3Dmol with synchronized controls
   - Upload your own PDB files or use example structures
   - Interactive visualization controls: Backbone trace, Sidechains
   - **pLDDT confidence analysis** with:
     - Color-coded 3D visualization (red=low to blue=high confidence)
     - Per-residue pLDDT line plot
     - pLDDT distribution histogram
     - Confidence summary statistics
   - Interactive molecular viewer with rotation, zoom, and pan controls
   - **Real evaluation data** from the paper including:
     - Actual generated protein sequences from 4 GO term sets
     - Real TM-scores (0.82-0.98) and pLDDT values
     - Actual PDB structures from ESMFold predictions
     - Ground truth structures matched to predictions

## Key Concepts

### Gene Ontology (GO) Terms
Standardized descriptors for protein properties and functions, enabling precise specification of desired characteristics.

### Reference-Guided Generation
Using existing protein sequences as templates to guide the generation of new sequences with similar properties.

### Top-TM-Score Evaluation
Structural similarity metric that prioritizes closest-shape matching with ground truth proteins.

## Paper Citation

```
@article{tan2024progo,
  title={PRO-GO: Reference-Guided Protein Sequence Generation using Gene Ontology Terms},
  author={Tan, Darren and McLoughlin, Ian and Ng, Aik Beng and Wang, Zhengkui and Stern, Abraham C and See, Simon},
  journal={IEEE Access},
  year={2024},
  doi={10.1109/ACCESS.2024.0429000}
}
```

## Authors

- Darren Tan (NVIDIA Singapore, Singapore Institute of Technology)
- Ian McLoughlin (Singapore Institute of Technology)
- Aik Beng Ng (NVIDIA Singapore)
- Zhengkui Wang (Singapore Institute of Technology)
- Abraham C Stern (NVIDIA Singapore)
- Simon See (NVIDIA Singapore)

## Support

This work was supported by the Singapore Economic Development Board and NVIDIA Singapore via the Industrial Postgraduate Program PhD scholarship.
