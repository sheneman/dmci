;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; DMCI: Compiling scheme into composable and
;;       differentiable neural network representations
;;
;; diffesm_s.scm: DiffESM-S: a 97-node Earth-system model in Scheme, a production-scale batching benchmark (Experiment H)
;;
;; Luke Sheneman
;; Research Computing and Data Services (RCDS)
;; Institute for Interdisciplinary Data Sciences (IIDS)
;; University of Idaho
;; sheneman@uidaho.edu
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; DiffESM-S: Differentiable Symbolic Earth System Model (Small)
;;;
;;; A coupled climate-carbon-biosphere model for stress-testing
;;; Differentiable Meta-Circular Interpretation (DMCI).
;;;
;;; Inspired by FaIR (emissions-to-temperature) and CLIMBER-X
;;; (multi-reservoir carbon cycle), but designed as a differentiable
;;; symbolic benchmark rather than a production climate code.
;;;
;;; State variables:  20  (carbon, temperature, ice, aerosol, biosphere)
;;; Learnable params: 70  (rate constants, sensitivities, feedbacks)
;;; Emission inputs:   4  (CO2, CH4, N2O, SO2 per timestep)
;;; Total inputs:      95  (1 control + 20 initial states + 4 emissions + 70 params)
;;; Timesteps:         configurable via n_steps input
;;;
;;; The model is fully batchable (no heap/list ops) and returns
;;; the final surface temperature anomaly after n_steps annual steps.
;;; All 70 parameters produce non-zero gradients.
;;;
;;; Compile with:
;;;   graph = compile_scheme(source, inputs={...all 95 inputs...})
;;;   result = evaluate_batched(graph, input_dict)
;;;   result.backward()  # gradients flow through all n_steps
;;;
;;; Recommended default parameters produce stable integration for 200+ years:
;;;   Ts(10yr) ≈ -0.63 K, Ts(100yr) ≈ -0.26 K, Ts(200yr) ≈ +0.84 K
;;;   (initial aerosol cooling → gradual CO2-driven warming)

;; =====================================================================
;; Main simulation loop
;; =====================================================================
;; Loop parameters: 2 control + 20 state = 22 total
;; Captured from outer scope: 4 emissions + 70 parameters = 74 free vars

(loop
  ;; --- control ---
  ((n n_steps)
   (year 0.0)
   ;; --- carbon cycle (GtC) ---
   (C_atm     C_atm_0)       ; atmospheric CO2 pool
   (C_upper   C_upper_0)     ; upper ocean dissolved carbon
   (C_deep    C_deep_0)      ; deep ocean carbon
   (C_veg     C_veg_0)       ; vegetation biomass carbon
   (C_soilf   C_soilf_0)     ; fast-turnover soil carbon
   (C_soils   C_soils_0)     ; slow-turnover soil carbon
   (C_perm    C_perm_0)      ; permafrost carbon
   ;; --- methane (ppb) ---
   (CH4       CH4_0)         ; atmospheric methane
   ;; --- nitrous oxide (ppb) ---
   (N2O       N2O_0)         ; atmospheric N2O
   ;; --- temperature (K anomaly) ---
   (Ts        Ts_0)          ; surface temperature anomaly
   (Td        Td_0)          ; deep ocean temperature anomaly
   ;; --- cryosphere ---
   (ice       ice_0)         ; polar ice fraction (0 to 1)
   ;; --- aerosols (Tg) ---
   (sulf      sulf_0)        ; sulfate aerosol burden
   (bca       bca_0)         ; black carbon aerosol burden
   ;; --- land biosphere ---
   (b1_l      b1_l_0)        ; tropical forest leaf area index
   (b1_w      b1_w_0)        ; tropical forest wood carbon (GtC)
   (b2_l      b2_l_0)        ; temperate forest LAI
   (b2_w      b2_w_0)        ; temperate forest wood carbon
   (b3_l      b3_l_0)        ; grassland/shrub LAI
   (b3_w      b3_w_0))       ; grassland/shrub root carbon

  ;; --- loop body ---
  (if (= n 0)
    Ts    ; return final surface temperature anomaly

    (let*
      ;; =====================================================================
      ;; MODULE 1: Radiative Forcing
      ;; =====================================================================
      ;; CO2: logarithmic forcing (Myhre et al. 1998)
      ((RF_co2  (* p_rf_co2 (log (/ C_atm p_rf_co2r))))

       ;; CH4: square-root forcing
       (RF_ch4  (* p_rf_ch4 (- (sqrt CH4) (sqrt p_rf_ch4r))))

       ;; N2O: square-root forcing
       (RF_n2o  (* p_rf_n2o (- (sqrt N2O) (sqrt p_rf_n2or))))

       ;; Aerosol forcing: direct (sulfate negative, BC positive) + indirect
       (RF_aero (+ (* (- 0.0 p_rf_sulf) sulf)
                   (* p_rf_bc bca)
                   (* p_rf_ind (log (+ 1.0 sulf)))))

       ;; Ice-albedo forcing: less ice = lower albedo = positive forcing
       (albedo_eff  (+ (* p_a_ice ice) (* p_a_ocn (- 1.0 ice))))
       (RF_albedo   (* p_rf_alb (- p_a_ref albedo_eff)))

       ;; Total forcing with feedback amplification
       (RF_total (* (+ RF_co2 RF_ch4 RF_n2o RF_aero RF_albedo p_rf_vol)
                    p_rf_fb))

       ;; =====================================================================
       ;; MODULE 2: Two-Layer Temperature (Geoffroy et al. 2013)
       ;; =====================================================================
       ;; Surface layer: dTs/dt = (F - lambda*Ts - gamma*(Ts-Td)) / Cm
       (dTs  (/ (- RF_total (+ (* p_lam Ts) (* p_kht (- Ts Td))))
               p_Cm))

       ;; Deep ocean: dTd/dt = gamma*(Ts-Td) / Cd
       (dTd  (/ (* p_kht (- Ts Td)) p_Cd))

       ;; Updated temperatures
       (Ts_new  (+ Ts dTs))
       (Td_new  (+ Td dTd))

       ;; =====================================================================
       ;; MODULE 3: Carbon Cycle
       ;; =====================================================================
       ;; CO2 fertilization: NPP increases logarithmically with CO2
       (npp_co2  (+ 1.0 (* p_cf (log (/ C_atm 590.0)))))

       ;; Temperature effect on NPP: Gaussian around optimum
       (npp_tdiff  (- Ts_new p_npp_to))
       (npp_temp   (exp (/ (- 0.0 (* npp_tdiff npp_tdiff)) 10.0)))

       ;; Aggregate net primary production
       (NPP  (* p_npp npp_co2 npp_temp))

       ;; Temperature effect on respiration (Q10-like)
       (resp_f  (exp (* p_tr Ts_new)))

       ;; Vegetation respiration
       (resp_veg  (* p_resp C_veg resp_f))

       ;; Litterfall: vegetation to fast soil pool
       (litter  (* p_lit C_veg))

       ;; Soil decomposition (temperature-dependent)
       (decomp_fast  (* p_df C_soilf resp_f))
       (decomp_slow  (* p_ds C_soils resp_f))

       ;; Fast soil to slow soil transfer
       (fast2slow  (* p_fs C_soilf))

       ;; Permafrost thaw: sigmoid response to temperature
       ;; Smooth threshold avoids gradient-killing hard branch
       (thaw_sigmoid  (/ 1.0 (+ 1.0 (exp (* (- 0.0 p_pts) (- Ts_new 2.0))))))
       (perm_release  (* p_pt thaw_sigmoid C_perm))

       ;; Ocean-atmosphere carbon exchange
       ;; Uptake decreases with warming (solubility pump)
       (ocean_uptake  (* p_ao (- C_atm (* p_ots Ts_new C_upper))))
       (ocean_outgas  (* p_oa C_upper))

       ;; Upper-deep ocean exchange
       (ocean_deep_xfer  (- (* p_od C_upper) (* p_do C_deep)))

       ;; =====================================================================
       ;; MODULE 4: Methane Cycle
       ;; =====================================================================
       ;; OH sink: lifetime decreases with warming
       (ch4_sink  (* p_md (exp (* p_mts Ts_new)) CH4))

       ;; Wetland emissions: increase with temperature
       (ch4_wet   (* p_mw (exp (* 0.05 Ts_new))))

       ;; Permafrost methane release
       (ch4_perm  (* p_mp thaw_sigmoid C_perm))

       ;; Methane budget (p_mo = ocean flux, p_mnb = natural baseline)
       (dCH4  (+ eCH4 p_mnb ch4_wet p_mo ch4_perm (- 0.0 ch4_sink)))
       (CH4_new  (+ CH4 dCH4))

       ;; =====================================================================
       ;; MODULE 5: Nitrous Oxide Cycle
       ;; =====================================================================
       ;; Soil emissions (temperature-dependent)
       (n2o_soil  (* p_ns (exp (* p_nts Ts_new))))

       ;; Stratospheric sink
       (n2o_sink  (* p_nd N2O))

       ;; N2O budget (p_no = ocean source)
       (dN2O  (+ eN2O n2o_soil p_no (- 0.0 n2o_sink)))
       (N2O_new  (+ N2O dN2O))

       ;; =====================================================================
       ;; MODULE 6: Ice / Albedo Feedback
       ;; =====================================================================
       ;; Equilibrium ice fraction: smooth sigmoid transition
       (ice_eq  (/ 1.0 (+ 1.0 (exp (* p_im (- Ts_new p_it))))))

       ;; Relaxation dynamics toward equilibrium
       (d_ice  (* p_ir (- ice_eq ice)))
       (ice_raw  (+ ice d_ice))

       ;; Smooth clamp to (0, 1): sigmoid squashing
       (ice_new  (/ 1.0 (+ 1.0 (exp (* (- 0.0 20.0) (- ice_raw 0.5))))))

       ;; =====================================================================
       ;; MODULE 7: Aerosols
       ;; =====================================================================
       ;; Sulfate: production from SO2, deposition increases with warming
       (sulf_prod  (* p_sf eSO2))
       (sulf_dep   (* p_sd sulf (+ 1.0 (* p_alt Ts_new))))
       (d_sulf     (- sulf_prod sulf_dep))
       (sulf_new   (+ sulf d_sulf))

       ;; Black carbon: correlates with fossil fuel use
       (bc_prod  (* p_be eCO2))
       (bc_dep   (* p_bd bca))
       (d_bc     (- bc_prod bc_dep))
       (bca_new  (+ bca d_bc))

       ;; =====================================================================
       ;; MODULE 8: Land Biosphere (3 biomes)
       ;; =====================================================================

       ;; --- Biome 1: Tropical forest ---
       ;; Gross photosynthesis (GtC/yr); 5% allocated to leaf, 25% to wood
       (b1_photo  (* p_b1p b1_l npp_co2
                     (exp (/ (- 0.0 (* (- Ts_new p_b1to) (- Ts_new p_b1to)))
                             20.0))))
       (b1_lleaf  (* p_b1lt b1_l))
       (b1_lwood  (* p_b1wt b1_w))
       (b1_wstr   (* p_b1ws b1_l
                     (/ 1.0 (+ 1.0 (exp (* (- 0.0 2.0) (- Ts_new 3.0)))))))
       (d_b1_l    (- (* 0.05 b1_photo) b1_lleaf b1_wstr))
       (d_b1_w    (- (* 0.25 b1_photo) b1_lwood))
       (b1_l_new  (+ b1_l d_b1_l))
       (b1_w_new  (+ b1_w d_b1_w))

       ;; --- Biome 2: Temperate forest ---
       (b2_photo  (* p_b2p b2_l npp_co2
                     (exp (/ (- 0.0 (* (- Ts_new p_b2to) (- Ts_new p_b2to)))
                             20.0))))
       (b2_lleaf  (* p_b2lt b2_l))
       (b2_lwood  (* p_b2wt b2_w))
       (b2_wstr   (* p_b2ws b2_l
                     (/ 1.0 (+ 1.0 (exp (* (- 0.0 2.0) (- Ts_new 3.0)))))))
       (b2_frost  (* p_b2fs b2_l
                     (/ 1.0 (+ 1.0 (exp (* 5.0 Ts_new))))))
       (d_b2_l    (- (* 0.05 b2_photo) b2_lleaf b2_wstr b2_frost))
       (d_b2_w    (- (* 0.25 b2_photo) b2_lwood))
       (b2_l_new  (+ b2_l d_b2_l))
       (b2_w_new  (+ b2_w d_b2_w))

       ;; --- Biome 3: Grassland / shrubland ---
       (b3_photo  (* p_b3p b3_l npp_co2
                     (exp (/ (- 0.0 (* (- Ts_new p_b3to) (- Ts_new p_b3to)))
                             20.0))))
       (b3_lleaf  (* p_b3lt b3_l))
       (b3_lwood  (* p_b3wt b3_w))
       (b3_wstr   (* p_b3ws b3_l
                     (/ 1.0 (+ 1.0 (exp (* (- 0.0 2.0) (- Ts_new 3.0)))))))
       (b3_frost  (* p_b3fs b3_l
                     (/ 1.0 (+ 1.0 (exp (* 5.0 Ts_new))))))
       (d_b3_l    (- (* 0.05 b3_photo) b3_lleaf b3_wstr b3_frost))
       (d_b3_w    (- (* 0.25 b3_photo) b3_lwood))
       (b3_l_new  (+ b3_l d_b3_l))
       (b3_w_new  (+ b3_w d_b3_w))

       ;; =====================================================================
       ;; MODULE 9: Carbon Pool Updates (after biosphere, for gradient flow)
       ;; =====================================================================
       ;; NPP = main vegetation CO2 sink; biome photo = additional land use flux
       ;; Wood decomposition returns carbon to atmosphere
       (dC_atm   (+ eCO2 (- 0.0 NPP b1_photo b2_photo b3_photo)
                    b1_lwood b2_lwood b3_lwood
                    resp_veg decomp_fast decomp_slow
                    perm_release (- 0.0 ocean_uptake) ocean_outgas))
       (dC_upper (- ocean_uptake ocean_outgas ocean_deep_xfer))
       (dC_deep  ocean_deep_xfer)
       (dC_veg   (- NPP resp_veg litter))
       (dC_soilf (- litter decomp_fast fast2slow))
       (dC_soils (- fast2slow decomp_slow))
       (dC_perm  (- 0.0 perm_release))

       (C_atm_new   (+ C_atm dC_atm))
       (C_upper_new (+ C_upper dC_upper))
       (C_deep_new  (+ C_deep dC_deep))
       (C_veg_new   (+ C_veg dC_veg))
       (C_soilf_new (+ C_soilf dC_soilf))
       (C_soils_new (+ C_soils dC_soils))
       (C_perm_new  (+ C_perm dC_perm)))

      ;; === Recur with all 22 updated loop variables ===
      (recur
        (- n 1)
        (+ year 1.0)
        ;; carbon
        C_atm_new C_upper_new C_deep_new C_veg_new
        C_soilf_new C_soils_new C_perm_new
        ;; methane
        CH4_new
        ;; N2O
        N2O_new
        ;; temperature
        Ts_new Td_new
        ;; ice
        ice_new
        ;; aerosols
        sulf_new bca_new
        ;; biosphere
        b1_l_new b1_w_new
        b2_l_new b2_w_new
        b3_l_new b3_w_new))))
