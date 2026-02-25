# ══════════════════════════════════════════════════════════════
# TAB — COMBINÉ
# ══════════════════════════════════════════════════════════════
with tab_comb:
    import math as _math
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div class="card-title" style="margin-bottom:6px;">🎰 Constructeur de Combiné</div>
        <div style="font-size:0.82rem; color:#4a5e60;">
            Ajoute tes sélections · l'app calcule la proba réelle et te dit si le combiné vaut le coup
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if atp_data is None:
        st.error("Données ATP non disponibles.")
    else:
        # ── Chargement global des cotes (inchangé) ───────────
        cb_col1, cb_col2 = st.columns([2, 3])
        with cb_col1:
            if st.button("🔍 Charger toutes les cotes live", key="comb_load_odds",
                         help="Une seule requête API pour tous tes matchs — économise le quota"):
                with st.spinner("Chargement des cotes ATP en cours..."):
                    global_odds = fetch_all_atp_odds()
                st.session_state["odds_global_index"] = global_odds
                if global_odds.get("ok"):
                    n_t = global_odds.get("total", 0)
                    cr = global_odds.get("credits_remaining", "?")
                    keys = global_odds.get("sport_keys", [])
                    st.success(
                        f"✅ {n_t} matchs trouvés · {len(keys)} tournois ATP "
                        f"· {cr} crédits restants"
                    )
                else:
                    err = global_odds.get("error", "")
                    st.warning(
                        "⚠️ API inaccessible depuis Streamlit Cloud. "
                        "Cause probable : restrictions réseau du serveur. "
                        "Entre les cotes manuellement."
                    )
                    with st.expander("Détail erreur"):
                        st.code(err)
        
        with cb_col2:
            gidx = st.session_state.get("odds_global_index", {})
            if gidx.get("ok"):
                n_ev = gidx.get("total", 0)
                cr = gidx.get("credits_remaining", "?")
                st.markdown(
                    f'<div style="color:#3dd68c; font-size:0.8rem; padding-top:10px;">' +
                    f'✅ {n_ev} matchs · {cr} crédits restants</div>',
                    unsafe_allow_html=True
                )
            elif gidx.get("ok") == False:
                st.markdown(
                    '<div style="color:#e07878; font-size:0.8rem; padding-top:10px;">' +
                    '⚠️ API indisponible — saisie manuelle</div>',
                    unsafe_allow_html=True
                )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # ── Paramètres combiné ────────────────────────────────
        n_comb = st.number_input("Nombre de sélections", min_value=2, max_value=20, value=4, step=1, key="comb_n")
        mise_comb = st.number_input("Mise (€)", min_value=0.10, max_value=1000.0, value=1.0, step=0.10, key="comb_mise", format="%.2f")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # ── Saisie des sélections + cote manuelle ─────────────
        selections = []
        global_idx = st.session_state.get("odds_global_index", {}).get("index", {})
        
        for ci in range(int(n_comb)):
            st.markdown(f'<div style="font-size:0.7rem; color:#3dd68c; letter-spacing:3px; text-transform:uppercase; margin-bottom:8px;">Sélection {ci+1}</div>', unsafe_allow_html=True)
            
            sc1, sc2, sc3 = st.columns([2, 2, 3])
            with sc1:
                j1_c = st.selectbox("Joueur 1", atp_player_list, key=f"comb_j1_{ci}",
                                    index=None, placeholder="Joueur 1...")
            with sc2:
                j2_opts = [p for p in atp_player_list if p != j1_c] if j1_c else atp_player_list
                j2_c = st.selectbox("Joueur 2", j2_opts, key=f"comb_j2_{ci}",
                                    index=None, placeholder="Joueur 2...")
            with sc3:
                tourn_c = st.selectbox("Tournoi", TOURN_NAMES, key=f"comb_tourn_{ci}")
            
            surf_c, level_c, bo_c = TOURN_DICT.get(tourn_c, ("Hard","A",3))
            sc_color = {"Hard":"#4a90d9","Clay":"#c8703a","Grass":"#3dd68c"}.get(surf_c,"#4a5e60")
            
            # Sélection du joueur + cote manuelle
            oc1, oc2, oc3 = st.columns([3, 2, 1])
            with oc1:
                sel_player = st.selectbox(
                    "Jouer sur", 
                    [j1_c, j2_c] if j1_c and j2_c else ["—"],
                    key=f"comb_sel_{ci}", 
                    label_visibility="collapsed"
                ) if j1_c and j2_c else None
            
            with oc2:
                cote_c = st.text_input(
                    "Cote du joueur sélectionné", 
                    key=f"comb_cote_{ci}",
                    placeholder="ex: 1.45 ou 2.10",
                    help="Saisis la cote du joueur que tu choisis (décimale, point ou virgule acceptée)"
                )
            
            with oc3:
                st.markdown(f'<div style="background:{sc_color}22; color:{sc_color}; border:1px solid {sc_color}44; padding:6px 10px; border-radius:8px; font-size:0.72rem; text-align:center; margin-top:28px;">{surf_c}</div>', unsafe_allow_html=True)
            
            # Conversion cote en float (gère virgule/point)
            cote_val = None
            if cote_c:
                try:
                    cote_val = float(cote_c.replace(",", "."))
                    if cote_val <= 1.0:
                        cote_val = None
                except:
                    cote_val = None
            
            selections.append({
                "j1": j1_c, 
                "j2": j2_c, 
                "joueur": sel_player,
                "surface": surf_c, 
                "level": level_c, 
                "best_of": bo_c,
                "cote": cote_val, 
                "tournoi": tourn_c,
            })
            
            if ci < int(n_comb)-1:
                st.markdown('<div style="border-top:1px solid #1a2a2c; margin:12px 0;"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # ── Bouton analyser ───────────────────────────────────
        col_comb_btn = st.columns([1,2,1])
        with col_comb_btn[1]:
            comb_clicked = st.button("⚡ ANALYSER LE COMBINÉ", use_container_width=True, key="comb_btn")
        
        if comb_clicked:
            # Filtrer sélections valides
            valid_sels = [s for s in selections if s["j1"] and s["j2"] and s["joueur"] and s["joueur"] != "—"]
            if len(valid_sels) < 2:
                st.warning("Renseigne au moins 2 sélections complètes.")
            else:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # ── Calcul proba + cote pour chaque sélection ─
                results_comb = []
                model_cache_c, scaler_cache_c = {}, {}
                
                for s in valid_sels:
                    if s["surface"] not in model_cache_c:
                        model_cache_c[s["surface"]] = load_model("atp", s["surface"])
                        scaler_cache_c[s["surface"]] = load_scaler("atp", s["surface"])
                    
                    model_c = model_cache_c[s["surface"]]
                    scaler_c = scaler_cache_c[s["surface"]]
                    
                    s1_c = get_player_stats(atp_data, s["j1"], s["surface"])
                    s2_c = get_player_stats(atp_data, s["j2"], s["surface"])
                    h2h_c = get_h2h(atp_data, s["j1"], s["j2"], s["surface"])
                    
                    proba_c = None
                    if model_c and s1_c and s2_c:
                        try:
                            n_c = model_c.input_shape[-1]
                        except:
                            n_c = 26
                        
                        fv_c = build_feature_vector(s1_c, s2_c, h2h_c["h2h_score"],
                                                    s["surface"], float(s["best_of"]),
                                                    s["level"], n_features=n_c)
                        X_c = np.array(fv_c).reshape(1,-1)
                        if scaler_c and getattr(scaler_c, "n_features_in_", 0) == X_c.shape[1]:
                            X_c = scaler_c.transform(X_c)
                        
                        raw = float(model_c.predict(X_c, verbose=0)[0][0])
                        proba_c = raw if s["joueur"] == s["j1"] else 1 - raw
                    else:
                        proba_c = 0.55  # fallback
                    
                    results_comb.append({
                        **s,
                        "proba_model": proba_c,
                    })
                
                # ── Affichage récapitulatif avec cotes manuelles ────────
                st.markdown("""
                <div style="font-size:0.72rem; color:#4a5e60; letter-spacing:2px;
                            text-transform:uppercase; margin-bottom:12px;">Récapitulatif des sélections</div>
                """, unsafe_allow_html=True)
                
                for r in results_comb:
                    cote_txt = f"{r['cote']:.2f}" if r['cote'] else "— (saisie manuelle requise)"
                    st.markdown(
                        f'<div style="display:flex; align-items:center; gap:16px; padding:12px; background:#111a1c; border-radius:10px; margin-bottom:8px;">'
                        f'<div style="flex:1;"><strong>{r["joueur"]}</strong><br><small>{r["j1"]} vs {r["j2"]} · {r["tournoi"]}</small></div>'
                        f'<div style="min-width:80px; text-align:center;">'
                        f'<div style="font-size:0.65rem; color:#4a5e60;">PROBA</div>'
                        f'<div style="font-size:1.4rem; color:#3dd68c;">{r["proba_model"]:.0%}</div>'
                        f'</div>'
                        f'<div style="min-width:100px; text-align:center;">'
                        f'<div style="font-size:0.65rem; color:#4a5e60;">COTE SAISIE</div>'
                        f'<div style="font-size:1.4rem; color:#c8c0b0;">{cote_txt}</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # ── Calcul global ────────────────────────────────
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                proba_globale = 1.0
                cote_globale = 1.0
                all_cotes_saisies = True
                
                for r in results_comb:
                    proba_globale *= r["proba_model"]
                    if r["cote"]:
                        cote_globale *= r["cote"]
                    else:
                        all_cotes_saisies = False
                
                gain_potentiel = mise_comb * cote_globale if all_cotes_saisies else None
                esperance = proba_globale * gain_potentiel - mise_comb if all_cotes_saisies else None
                
                # Affichage résultat
                sg1, sg2, sg3 = st.columns(3)
                with sg1:
                    st.metric("Probabilité réelle", f"{proba_globale:.1%}")
                with sg2:
                    if all_cotes_saisies:
                        st.metric("Cote combinée", f"{cote_globale:.2f}")
                    else:
                        st.metric("Cote combinée", "—", delta="Saisis toutes les cotes")
                with sg3:
                    if all_cotes_saisies:
                        st.metric("Espérance", f"{esperance:+.2f} €" if esperance is not None else "—")
                    else:
                        st.metric("Espérance", "—", delta="Manque cotes")
                
                if not all_cotes_saisies:
                    st.info("Pour voir la cote combinée et l'espérance, saisis la cote de chaque sélection ci-dessus.")
