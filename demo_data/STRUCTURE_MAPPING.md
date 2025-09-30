# Structure Correspondence for New Demo Sets

## Predicted Structure to Ground Truth Mapping

Based on TM-score analysis from paper2_evaluation:

## Set 005 - Preventing Runaway Cell Division

| Predicted | Original Sequence | Best Match | TM-Score |
|-----------|------------------|------------|----------|
| set005_predicted_1.pdb | Sequence_29 | 9BI4 | 0.8417 |
| set005_predicted_2.pdb | Sequence_2 | 8XVT | 0.8326 |
| set005_predicted_3.pdb | Sequence_79 | 8XVT | 0.8318 |
| set005_predicted_4.pdb | Sequence_98 | 5OAF | 0.8273 |
| set005_predicted_5.pdb | Sequence_38 | 7OLE | 0.8229 |

**Available ground truth**: 9BI4, 8XVT, 5OAF, 7OLE (4 unique structures)
- Note: 2 sequences converge to 8XVT

## Set 051 - Rebalancing Cellular Energy and Metabolism

| Predicted | Original Sequence | Best Match | TM-Score |
|-----------|------------------|------------|----------|
| set051_predicted_1.pdb | Sequence_1 | N/A | N/A |
| set051_predicted_2.pdb | Sequence_2 | N/A | N/A |
| set051_predicted_3.pdb | Sequence_3 | N/A | N/A |
| set051_predicted_4.pdb | Sequence_4 | N/A | N/A |
| set051_predicted_5.pdb | Sequence_5 | N/A | N/A |

**Note**: No TM-score analysis available. Using first 5 sequences.

## Set 060 - Controlling Drug and Ion Flow

| Predicted | Original Sequence | Best Match | TM-Score |
|-----------|------------------|------------|----------|
| set060_predicted_1.pdb | Sequence_68 | 5LK7 | 0.9325 |
| set060_predicted_2.pdb | Sequence_78 | 6EH1 | 0.9275 |
| set060_predicted_3.pdb | Sequence_76 | 5LK7 | 0.9242 |
| set060_predicted_4.pdb | Sequence_34 | 6EGX | 0.9240 |
| set060_predicted_5.pdb | Sequence_99 | 5LSF | 0.9218 |

**Available ground truth**: 5LK7, 6EH1, 6EGX, 5LSF (4 unique structures)
- Note: 2 sequences converge to 5LK7

## Set 076 - Targeting Uncontrolled Cell Growth (Kinase Signaling)

| Predicted | Original Sequence | Best Match | TM-Score |
|-----------|------------------|------------|----------|
| set076_predicted_1.pdb | Sequence_48 | 3TN8 | 0.9527 |
| set076_predicted_2.pdb | Sequence_77 | 6C9H | 0.9522 |
| set076_predicted_3.pdb | Sequence_33 | 2A1A | 0.9438 |
| set076_predicted_4.pdb | Sequence_90 | 5HVJ | 0.9317 |
| set076_predicted_5.pdb | Sequence_56 | 7MN5 | 0.9301 |

**Available ground truth**: All 5 unique structures present
- Exceptional performance with all TM-scores >93%

## Set 088 - Rebalancing Cellular Energy (Biosynthetic)

| Predicted | Original Sequence | Best Match | TM-Score |
|-----------|------------------|------------|----------|
| set088_predicted_1.pdb | Sequence_46 | 4MRT | 0.9858 |
| set088_predicted_2.pdb | Sequence_71 | 8P5O | 0.9806 |
| set088_predicted_3.pdb | Sequence_85 | 4MRT | 0.9789 |
| set088_predicted_4.pdb | Sequence_86 | 4MRT | 0.9786 |
| set088_predicted_5.pdb | Sequence_6 | 4MRT | 0.9751 |

**Available ground truth**: 4MRT, 8P5O (2 unique structures)
- Note: 4 out of 5 sequences converge to 4MRT!
- Outstanding performance with all TM-scores >97.5%

## Key Insights

1. **Convergence patterns vary by protein family**:
   - Set 088: Strong convergence (4/5 → 4MRT)
   - Set 076: No convergence (all different structures)
   - Sets 005 & 060: Partial convergence (2/5 to same structure)

2. **Performance correlates with protein type**:
   - Biosynthetic proteins (088): >97.5% similarity
   - Kinases (076) & Transporters (060): >92% similarity
   - Cell cycle proteins (005): >82% similarity

3. **Ground truth availability**:
   - Some sets have fewer unique ground truth structures due to convergence
   - This actually demonstrates that AI captures essential family features
