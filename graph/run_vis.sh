dir_paths=(
# 'mono_rbphjorth_pli_None'
# 'mono_rbphjorth_plv_None'
# 'mono_rbphjorth_corr_None'
# 'mono_rbphjorth_corr_alpha'
# 'mono_rbphjorth_plv_alpha'
# 'mono_rbphjorth_pli_alpha'
# 'mono_rbphjorth_coherence_None'
# 'mono_rbphjorth_coherence_alpha'

# 'bi23_rbphjorth_pli_None'
# 'bi23_rbphjorth_plv_None'
# 'bi23_rbphjorth_corr_None'
# 'bi23_rbphjorth_corr_alpha'
# 'bi23_rbphjorth_plv_alpha'
# 'bi23_rbphjorth_pli_alpha'
# 'bi23_rbphjorth_coherence_None'
# 'bi23_rbphjorth_coherence_alpha'

'mono_rbphjorth_plv_None_8_4'
'mono_rbphjorth_pli_None_8_4'
'mono_rbphjorth_plv_alpha_8_4'
'mono_rbphjorth_coherence_None_8_4'
'mono_rbphjorth_coherence_alpha_8_4'
'mono_rbphjorth_pli_alpha_8_4'

'mono_rbphjorth_plv_alpha_1_0'
'mono_rbphjorth_coherence_alpha_1_0'
'mono_rbphjorth_coherence_None_1_0'
'mono_rbphjorth_plv_None_1_0'
'mono_rbphjorth_pli_None_1_0'
'mono_rbphjorth_pli_alpha_1_0'


'mono_rbphjorth_plv_None_8_7'
'mono_rbphjorth_pli_alpha_8_7'
'mono_rbphjorth_pli_None_8_7'
'mono_rbphjorth_plv_alpha_8_7'
'mono_rbphjorth_coherence_alpha_8_7'
'mono_rbphjorth_coherence_None_8_7'

'bi23_rbphjorth_coherence_None_1_0'
'bi23_rbphjorth_pli_None_1_0'
'bi23_rbphjorth_coherence_alpha_1_0'
'bi23_rbphjorth_plv_alpha_1_0'
'bi23_rbphjorth_plv_None_1_0'
'bi23_rbphjorth_pli_alpha_1_0'

)
SCRIPT="graph/vis.py"

for dir in "${dir_paths[@]}"; do
    python "$SCRIPT" \
        --pt_path "$dir" \
        --embedding_mode summary_only \
        --classifier_name logreg \
        --n_splits 5
done
