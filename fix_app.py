import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []
in_tab2 = False
in_tab3 = False

for i, line in enumerate(lines):
    if line.startswith('with tab2:'):
        in_tab2 = True
        out_lines.append(line)
        continue
    elif line.startswith('with tab3:'):
        in_tab2 = False
        in_tab3 = True
        out_lines.append(line)
        continue
    elif line.startswith('# ═══════════════════════════════════════════════════════════════') or line.startswith('# Footer'):
        in_tab3 = False
        out_lines.append(line)
        continue

    if in_tab2:
        if 'st.markdown("### Compare Driver Laps")' in line:
            out_lines.append(line)
            out_lines.append('    \n')
            out_lines.append('    if session_type_display != "Qualifying":\n')
            out_lines.append('        st.warning("Lap Analysis is currently only available for Qualifying sessions.")\n')
            out_lines.append('    else:\n')
            out_lines.append('        with st.form("lap_analysis_form"):\n')
            continue
        if 'col_select, col_telemetry = st.columns([1, 1])' in line:
            out_lines.append('            col_select, col_telemetry = st.columns([1, 1])\n')
            continue
        if 'with col_select:' in line:
            out_lines.append('            with col_select:\n')
            continue
        if 'selected_drivers = st.multiselect(' in line:
            out_lines.append('                safe_defaults = sorted(drv_list)[:2] if len(drv_list) >= 2 else sorted(drv_list)\n')
            out_lines.append('                selected_drivers = st.multiselect(\n')
            continue
        if 'default=sorted(drv_list)[:2] if len(drv_list) >= 2 else drv_list,' in line:
            out_lines.append('                    default=safe_defaults,\n')
            continue
        if 'with col_telemetry:' in line:
            out_lines.append('            with col_telemetry:\n')
            continue
        if 'if not selected_drivers:' in line:
            out_lines.append('            submit_lap = st.form_submit_button("Load Analysis")\n')
            out_lines.append('        \n')
            out_lines.append('        valid_drivers = [d for d in selected_drivers if d in drv_list]\n')
            out_lines.append('        if not valid_drivers:\n')
            continue
        if "max_selected_duration = max(driver_tel[drv]['Trel'].iloc[-1] for drv in selected_drivers)" in line:
            out_lines.append('            selected_drivers = valid_drivers\n')
            out_lines.append('            ' + line.lstrip())
            continue
            
        if not line.strip() == '':
            if 'st.markdown("### Compare Driver Laps")' not in line:
                if 'options=' in line or 'max_selections=' in line or 'default=' in line or '"Select drivers' in line or '"Telemetry channels' in line:
                    out_lines.append('    ' + line)
                else:
                    out_lines.append('    ' + line)
        else:
            out_lines.append(line)
            
    elif in_tab3:
        if 'st.markdown("Compare pace difference between two drivers throughout the lap.")' in line:
            out_lines.append(line)
            out_lines.append('    \n')
            out_lines.append('    if session_type_display != "Qualifying":\n')
            out_lines.append('        st.warning("Time Delta Analysis is currently only available for Qualifying sessions.")\n')
            out_lines.append('    else:\n')
            out_lines.append('        with st.form("time_delta_form"):\n')
            continue
        if 'col1, col2 = st.columns(2)' in line:
            out_lines.append('            col1, col2 = st.columns(2)\n')
            continue
        if 'with col1:' in line:
            out_lines.append('            with col1:\n')
            continue
        if 'driver1 = st.selectbox(' in line:
            out_lines.append('                driver1 = st.selectbox("Driver 1 (Reference)", options=sorted(drv_list), index=0, key="delta_drv1")\n')
            continue
        if 'with col2:' in line:
            out_lines.append('            with col2:\n')
            continue
        if 'remaining = [d for d in sorted(drv_list) if d != driver1]' in line:
            out_lines.append('                remaining = [d for d in sorted(drv_list) if d != driver1]\n')
            continue
        if 'driver2 = st.selectbox(' in line:
            out_lines.append('                driver2 = st.selectbox("Driver 2 (Comparison)", options=remaining, index=0 if remaining else None, key="delta_drv2")\n')
            continue
        if 'if driver1 and driver2 and driver1 != driver2:' in line:
            out_lines.append('            submit_delta = st.form_submit_button("Load Analysis")\n')
            out_lines.append('        \n')
            out_lines.append('        if driver1 in drv_list and driver2 in drv_list and driver1 != driver2:\n')
            continue
            
        if not line.strip() == '':
            if 'st.markdown("### Time Delta Analysis")' not in line:
                out_lines.append('    ' + line)
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
    else:
        out_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(out_lines)
