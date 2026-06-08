;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; DMCI: Compiling scheme into composable and
;;       differentiable neural network representations
;;
;; diffsoc_s.scm: DiffSoc-S: a 206-node urban political-economy simulator in Scheme, a production-scale batching benchmark (Experiment H)
;;
;; Luke Sheneman
;; Research Computing and Data Services (RCDS)
;; Institute for Interdisciplinary Data Sciences (IIDS)
;; University of Idaho
;; sheneman@uidaho.edu
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; DiffSoc-S: Differentiable Urban Political Economy Simulator (Small)
;;;
;;; A coupled agent-based model of households, firms, schools, banks,
;;; neighborhoods, housing markets, labor markets, and migration,
;;; with smooth differentiable institutional rules.
;;;
;;; Agent classes:      3  (low, mid, high income)
;;; Neighborhoods:      5  (downtown, 2 inner ring, 2 outer ring)
;;; Agent groups:      15  (class × neighborhood)
;;; State variables:   96  (60 household + 25 neighborhood + 6 labor + 2 bank + 3 macro)
;;; Learnable params:  87  (behavioral, institutional, structural)
;;; Exogenous drivers:  5  (growth, interest, investment, housing supply, immigration)
;;; Structural inputs: 15  (capacity, distance, transit per neighborhood)
;;; Total inputs:     204  (1 control + 96 initial + 87 params + 5 drivers + 15 structural)
;;; Timesteps:        configurable via n_steps (annual)
;;;
;;; Modules:
;;;   1. Labor market        — employment, wages, sector dynamics
;;;   2. Income              — earnings, transfers, taxation
;;;   3. Credit access       — sigmoid credit scoring per group
;;;   4. Housing costs       — ownership/rental blend, mortgage
;;;   5. Wealth dynamics     — saving, consumption, returns
;;;   6. Housing market      — demand, supply, price/rent adjustment
;;;   7. Schools & skill     — quality, funding, peer effects, human capital
;;;   8. Migration           — attractiveness, population-conserving flows
;;;   9. Neighborhood        — amenity, crime, spillovers
;;;  10. Macro & inequality  — GDP, inflation, Gini-like index
;;;
;;; Key feedback loops:
;;;   labor → income → credit → housing → neighborhood → school → skill → labor
;;;   income → wealth → ownership → price appreciation → wealth inequality
;;;   housing prices → migration → composition → schools → skill → future income
;;;
;;; Returns: composite urban welfare index
;;;   0.1×avg_price + 0.5×avg_school + avg_low_income + 0.001×gdp
;;;
;;; Compile with:
;;;   graph = compile_scheme(source, inputs={...all 239 inputs...})
;;;   result = evaluate_batched(graph, input_dict)
;;;   result.backward()  # gradients flow through all n_steps

;; =====================================================================
;; Main simulation loop: 98 state variables (96 state + 2 control)
;; Captured from outer scope: 120 params + 5 drivers + 15 structural
;; =====================================================================

(loop
  ;; --- control ---
  ((n n_steps)
   (year 0.0)

   ;; --- household: low income × 5 neighborhoods ---
   ;; (population, income, wealth, skill)
   (lo1_pop lo1_pop_0) (lo1_inc lo1_inc_0) (lo1_wth lo1_wth_0) (lo1_skl lo1_skl_0)
   (lo2_pop lo2_pop_0) (lo2_inc lo2_inc_0) (lo2_wth lo2_wth_0) (lo2_skl lo2_skl_0)
   (lo3_pop lo3_pop_0) (lo3_inc lo3_inc_0) (lo3_wth lo3_wth_0) (lo3_skl lo3_skl_0)
   (lo4_pop lo4_pop_0) (lo4_inc lo4_inc_0) (lo4_wth lo4_wth_0) (lo4_skl lo4_skl_0)
   (lo5_pop lo5_pop_0) (lo5_inc lo5_inc_0) (lo5_wth lo5_wth_0) (lo5_skl lo5_skl_0)

   ;; --- household: mid income × 5 neighborhoods ---
   (mi1_pop mi1_pop_0) (mi1_inc mi1_inc_0) (mi1_wth mi1_wth_0) (mi1_skl mi1_skl_0)
   (mi2_pop mi2_pop_0) (mi2_inc mi2_inc_0) (mi2_wth mi2_wth_0) (mi2_skl mi2_skl_0)
   (mi3_pop mi3_pop_0) (mi3_inc mi3_inc_0) (mi3_wth mi3_wth_0) (mi3_skl mi3_skl_0)
   (mi4_pop mi4_pop_0) (mi4_inc mi4_inc_0) (mi4_wth mi4_wth_0) (mi4_skl mi4_skl_0)
   (mi5_pop mi5_pop_0) (mi5_inc mi5_inc_0) (mi5_wth mi5_wth_0) (mi5_skl mi5_skl_0)

   ;; --- household: high income × 5 neighborhoods ---
   (hi1_pop hi1_pop_0) (hi1_inc hi1_inc_0) (hi1_wth hi1_wth_0) (hi1_skl hi1_skl_0)
   (hi2_pop hi2_pop_0) (hi2_inc hi2_inc_0) (hi2_wth hi2_wth_0) (hi2_skl hi2_skl_0)
   (hi3_pop hi3_pop_0) (hi3_inc hi3_inc_0) (hi3_wth hi3_wth_0) (hi3_skl hi3_skl_0)
   (hi4_pop hi4_pop_0) (hi4_inc hi4_inc_0) (hi4_wth hi4_wth_0) (hi4_skl hi4_skl_0)
   (hi5_pop hi5_pop_0) (hi5_inc hi5_inc_0) (hi5_wth hi5_wth_0) (hi5_skl hi5_skl_0)

   ;; --- neighborhood state ---
   ;; (price, rent, school quality, amenity, crime)
   (n1_prc n1_prc_0) (n1_rnt n1_rnt_0) (n1_sch n1_sch_0) (n1_amn n1_amn_0) (n1_crm n1_crm_0)
   (n2_prc n2_prc_0) (n2_rnt n2_rnt_0) (n2_sch n2_sch_0) (n2_amn n2_amn_0) (n2_crm n2_crm_0)
   (n3_prc n3_prc_0) (n3_rnt n3_rnt_0) (n3_sch n3_sch_0) (n3_amn n3_amn_0) (n3_crm n3_crm_0)
   (n4_prc n4_prc_0) (n4_rnt n4_rnt_0) (n4_sch n4_sch_0) (n4_amn n4_amn_0) (n4_crm n4_crm_0)
   (n5_prc n5_prc_0) (n5_rnt n5_rnt_0) (n5_sch n5_sch_0) (n5_amn n5_amn_0) (n5_crm n5_crm_0)

   ;; --- labor market (3 sectors) ---
   (s1_dem s1_dem_0) (s1_wge s1_wge_0)   ; service
   (s2_dem s2_dem_0) (s2_wge s2_wge_0)   ; professional
   (s3_dem s3_dem_0) (s3_wge s3_wge_0)   ; finance/tech

   ;; --- banking ---
   (crd crd_0) (irate irate_0)

   ;; --- macro ---
   (gdp gdp_0) (infl infl_0) (ineq ineq_0))

  ;; --- termination ---
  (if (= n 0)
    ;; Return composite urban welfare index
    (+ (* 0.1 (/ (+ n1_prc n2_prc n3_prc n4_prc n5_prc) 5.0))
       (* 0.5 (/ (+ n1_sch n2_sch n3_sch n4_sch n5_sch) 5.0))
       (/ (+ lo1_inc lo2_inc lo3_inc lo4_inc lo5_inc) 5.0)
       (* 0.001 gdp))

    (let*
      ;; =================================================================
      ;; PRECOMPUTE: Aggregate quantities
      ;; =================================================================
      ;; Total population per neighborhood
      ((tot_1  (+ lo1_pop mi1_pop hi1_pop))
       (tot_2  (+ lo2_pop mi2_pop hi2_pop))
       (tot_3  (+ lo3_pop mi3_pop hi3_pop))
       (tot_4  (+ lo4_pop mi4_pop hi4_pop))
       (tot_5  (+ lo5_pop mi5_pop hi5_pop))

       ;; Total population per class
       (tot_lo  (+ lo1_pop lo2_pop lo3_pop lo4_pop lo5_pop))
       (tot_mi  (+ mi1_pop mi2_pop mi3_pop mi4_pop mi5_pop))
       (tot_hi  (+ hi1_pop hi2_pop hi3_pop hi4_pop hi5_pop))

       ;; Weighted average income per neighborhood (for school funding)
       (avginc_1  (/ (+ (* lo1_pop lo1_inc) (* mi1_pop mi1_inc) (* hi1_pop hi1_inc)) (+ tot_1 0.1)))
       (avginc_2  (/ (+ (* lo2_pop lo2_inc) (* mi2_pop mi2_inc) (* hi2_pop hi2_inc)) (+ tot_2 0.1)))
       (avginc_3  (/ (+ (* lo3_pop lo3_inc) (* mi3_pop mi3_inc) (* hi3_pop hi3_inc)) (+ tot_3 0.1)))
       (avginc_4  (/ (+ (* lo4_pop lo4_inc) (* mi4_pop mi4_inc) (* hi4_pop hi4_inc)) (+ tot_4 0.1)))
       (avginc_5  (/ (+ (* lo5_pop lo5_inc) (* mi5_pop mi5_inc) (* hi5_pop hi5_inc)) (+ tot_5 0.1)))

       ;; Average income per class (for migration price sensitivity)
       (avginc_lo  (/ (+ (* lo1_pop lo1_inc) (* lo2_pop lo2_inc) (* lo3_pop lo3_inc)
                         (* lo4_pop lo4_inc) (* lo5_pop lo5_inc)) (+ tot_lo 0.1)))
       (avginc_mi  (/ (+ (* mi1_pop mi1_inc) (* mi2_pop mi2_inc) (* mi3_pop mi3_inc)
                         (* mi4_pop mi4_inc) (* mi5_pop mi5_inc)) (+ tot_mi 0.1)))
       (avginc_hi  (/ (+ (* hi1_pop hi1_inc) (* hi2_pop hi2_inc) (* hi3_pop hi3_inc)
                         (* hi4_pop hi4_inc) (* hi5_pop hi5_inc)) (+ tot_hi 0.1)))

       ;; City-wide averages (for migration reference point)
       (avg_prc  (/ (+ n1_prc n2_prc n3_prc n4_prc n5_prc) 5.0))
       (avg_sch  (/ (+ n1_sch n2_sch n3_sch n4_sch n5_sch) 5.0))
       (avg_amn  (/ (+ n1_amn n2_amn n3_amn n4_amn n5_amn) 5.0))
       (avg_crm  (/ (+ n1_crm n2_crm n3_crm n4_crm n5_crm) 5.0))

       ;; Job accessibility per neighborhood (transit-weighted)
       (jacc_1  (/ tran_1 (+ dist_1 1.0)))
       (jacc_2  (/ tran_2 (+ dist_2 1.0)))
       (jacc_3  (/ tran_3 (+ dist_3 1.0)))
       (jacc_4  (/ tran_4 (+ dist_4 1.0)))
       (jacc_5  (/ tran_5 (+ dist_5 1.0)))

       ;; =================================================================
       ;; MODULE 1: Labor Market
       ;; =================================================================
       ;; Employment probability: sigmoid(base + skill + demand - commute + transit + network)
       ;; Low income → sector 1 (service)
       (lo1_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl lo1_skl) (* p_emp_dem s1_dem)
                   (- 0.0 (* p_emp_cmt dist_1)) (* p_emp_tran tran_1) (* p_emp_net (/ lo1_pop (+ tot_1 0.1)))
                   (- 0.0 (* p_emp_disc 1.0))))))))
       (lo2_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl lo2_skl) (* p_emp_dem s1_dem)
                   (- 0.0 (* p_emp_cmt dist_2)) (* p_emp_tran tran_2) (* p_emp_net (/ lo2_pop (+ tot_2 0.1)))
                   (- 0.0 (* p_emp_disc 1.0))))))))
       (lo3_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl lo3_skl) (* p_emp_dem s1_dem)
                   (- 0.0 (* p_emp_cmt dist_3)) (* p_emp_tran tran_3) (* p_emp_net (/ lo3_pop (+ tot_3 0.1)))
                   (- 0.0 (* p_emp_disc 1.0))))))))
       (lo4_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl lo4_skl) (* p_emp_dem s1_dem)
                   (- 0.0 (* p_emp_cmt dist_4)) (* p_emp_tran tran_4) (* p_emp_net (/ lo4_pop (+ tot_4 0.1)))
                   (- 0.0 (* p_emp_disc 1.0))))))))
       (lo5_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl lo5_skl) (* p_emp_dem s1_dem)
                   (- 0.0 (* p_emp_cmt dist_5)) (* p_emp_tran tran_5) (* p_emp_net (/ lo5_pop (+ tot_5 0.1)))
                   (- 0.0 (* p_emp_disc 1.0))))))))

       ;; Mid income → sector 2 (professional)
       (mi1_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl mi1_skl) (* p_emp_dem s2_dem)
                   (- 0.0 (* p_emp_cmt dist_1)) (* p_emp_tran tran_1) (* p_emp_net (/ mi1_pop (+ tot_1 0.1)))))))))
       (mi2_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl mi2_skl) (* p_emp_dem s2_dem)
                   (- 0.0 (* p_emp_cmt dist_2)) (* p_emp_tran tran_2) (* p_emp_net (/ mi2_pop (+ tot_2 0.1)))))))))
       (mi3_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl mi3_skl) (* p_emp_dem s2_dem)
                   (- 0.0 (* p_emp_cmt dist_3)) (* p_emp_tran tran_3) (* p_emp_net (/ mi3_pop (+ tot_3 0.1)))))))))
       (mi4_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl mi4_skl) (* p_emp_dem s2_dem)
                   (- 0.0 (* p_emp_cmt dist_4)) (* p_emp_tran tran_4) (* p_emp_net (/ mi4_pop (+ tot_4 0.1)))))))))
       (mi5_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl mi5_skl) (* p_emp_dem s2_dem)
                   (- 0.0 (* p_emp_cmt dist_5)) (* p_emp_tran tran_5) (* p_emp_net (/ mi5_pop (+ tot_5 0.1)))))))))

       ;; High income → sector 3 (finance/tech)
       (hi1_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl hi1_skl) (* p_emp_dem s3_dem)
                   (- 0.0 (* p_emp_cmt dist_1)) (* p_emp_tran tran_1) (* p_emp_net (/ hi1_pop (+ tot_1 0.1)))))))))
       (hi2_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl hi2_skl) (* p_emp_dem s3_dem)
                   (- 0.0 (* p_emp_cmt dist_2)) (* p_emp_tran tran_2) (* p_emp_net (/ hi2_pop (+ tot_2 0.1)))))))))
       (hi3_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl hi3_skl) (* p_emp_dem s3_dem)
                   (- 0.0 (* p_emp_cmt dist_3)) (* p_emp_tran tran_3) (* p_emp_net (/ hi3_pop (+ tot_3 0.1)))))))))
       (hi4_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl hi4_skl) (* p_emp_dem s3_dem)
                   (- 0.0 (* p_emp_cmt dist_4)) (* p_emp_tran tran_4) (* p_emp_net (/ hi4_pop (+ tot_4 0.1)))))))))
       (hi5_emp  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ p_emp_base (* p_emp_skl hi5_skl) (* p_emp_dem s3_dem)
                   (- 0.0 (* p_emp_cmt dist_5)) (* p_emp_tran tran_5) (* p_emp_net (/ hi5_pop (+ tot_5 0.1)))))))))

       ;; Total employed per sector
       (emp_s1  (+ (* lo1_pop lo1_emp) (* lo2_pop lo2_emp) (* lo3_pop lo3_emp)
                   (* lo4_pop lo4_emp) (* lo5_pop lo5_emp)))
       (emp_s2  (+ (* mi1_pop mi1_emp) (* mi2_pop mi2_emp) (* mi3_pop mi3_emp)
                   (* mi4_pop mi4_emp) (* mi5_pop mi5_emp)))
       (emp_s3  (+ (* hi1_pop hi1_emp) (* hi2_pop hi2_emp) (* hi3_pop hi3_emp)
                   (* hi4_pop hi4_emp) (* hi5_pop hi5_emp)))

       ;; Wage adjustment: excess demand + inflation passthrough + productivity
       (d_s1_wge  (* p_wge_adj (+ (* 0.1 (- s1_dem emp_s1)) (* p_wge_inf infl) (* p_wge_prod s1_wge))))
       (d_s2_wge  (* p_wge_adj (+ (* 0.1 (- s2_dem emp_s2)) (* p_wge_inf infl) (* p_wge_prod s2_wge))))
       (d_s3_wge  (* p_wge_adj (+ (* 0.1 (- s3_dem emp_s3)) (* p_wge_inf infl) (* p_wge_prod s3_wge))))

       ;; Labor demand dynamics
       (d_s1_dem  (* p_sec_grow (+ e_grow (- 0.0 (* 0.01 (- s1_wge 30.0))))))
       (d_s2_dem  (* p_sec_grow (+ (* 1.5 e_grow) (- 0.0 (* 0.01 (- s2_wge 55.0))))))
       (d_s3_dem  (* p_sec_grow (+ (* 2.0 e_grow) (- 0.0 (* 0.005 (- s3_wge 115.0))))))

       ;; =================================================================
       ;; MODULE 2: Income
       ;; =================================================================
       ;; Gross income = employment × wage × (1 + skill premium) + transfers
       ;; Low income gets transfer payments; minimum wage floor via soft-max
       (lo1_ginc  (+ (* lo1_emp (+ s1_wge (* p_skl_prm lo1_skl s1_wge)) (+ 1.0 (* p_pol_mw 0.1))) p_inc_xfer))
       (lo2_ginc  (+ (* lo2_emp (+ s1_wge (* p_skl_prm lo2_skl s1_wge)) (+ 1.0 (* p_pol_mw 0.1))) p_inc_xfer))
       (lo3_ginc  (+ (* lo3_emp (+ s1_wge (* p_skl_prm lo3_skl s1_wge)) (+ 1.0 (* p_pol_mw 0.1))) p_inc_xfer))
       (lo4_ginc  (+ (* lo4_emp (+ s1_wge (* p_skl_prm lo4_skl s1_wge)) (+ 1.0 (* p_pol_mw 0.1))) p_inc_xfer))
       (lo5_ginc  (+ (* lo5_emp (+ s1_wge (* p_skl_prm lo5_skl s1_wge)) (+ 1.0 (* p_pol_mw 0.1))) p_inc_xfer))

       (mi1_ginc  (* mi1_emp (+ s2_wge (* p_skl_prm mi1_skl s2_wge))))
       (mi2_ginc  (* mi2_emp (+ s2_wge (* p_skl_prm mi2_skl s2_wge))))
       (mi3_ginc  (* mi3_emp (+ s2_wge (* p_skl_prm mi3_skl s2_wge))))
       (mi4_ginc  (* mi4_emp (+ s2_wge (* p_skl_prm mi4_skl s2_wge))))
       (mi5_ginc  (* mi5_emp (+ s2_wge (* p_skl_prm mi5_skl s2_wge))))

       (hi1_ginc  (* hi1_emp (+ s3_wge (* p_skl_prm hi1_skl s3_wge))))
       (hi2_ginc  (* hi2_emp (+ s3_wge (* p_skl_prm hi2_skl s3_wge))))
       (hi3_ginc  (* hi3_emp (+ s3_wge (* p_skl_prm hi3_skl s3_wge))))
       (hi4_ginc  (* hi4_emp (+ s3_wge (* p_skl_prm hi4_skl s3_wge))))
       (hi5_ginc  (* hi5_emp (+ s3_wge (* p_skl_prm hi5_skl s3_wge))))

       ;; =================================================================
       ;; MODULE 3: Credit Access
       ;; =================================================================
       ;; Sigmoid credit score: income-to-price + wealth-to-price - crime penalty
       (lo1_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ lo1_ginc (+ n1_prc 1.0)))
                   (* p_crd_wth (/ lo1_wth (+ n1_prc 1.0))) (- 0.0 (* p_crd_crm n1_crm))
                   (- 0.0 (* p_crd_bias 1.0))))))))
       (lo2_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ lo2_ginc (+ n2_prc 1.0)))
                   (* p_crd_wth (/ lo2_wth (+ n2_prc 1.0))) (- 0.0 (* p_crd_crm n2_crm))
                   (- 0.0 (* p_crd_bias 1.0))))))))
       (lo3_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ lo3_ginc (+ n3_prc 1.0)))
                   (* p_crd_wth (/ lo3_wth (+ n3_prc 1.0))) (- 0.0 (* p_crd_crm n3_crm))
                   (- 0.0 (* p_crd_bias 1.0))))))))
       (lo4_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ lo4_ginc (+ n4_prc 1.0)))
                   (* p_crd_wth (/ lo4_wth (+ n4_prc 1.0))) (- 0.0 (* p_crd_crm n4_crm))
                   (- 0.0 (* p_crd_bias 1.0))))))))
       (lo5_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ lo5_ginc (+ n5_prc 1.0)))
                   (* p_crd_wth (/ lo5_wth (+ n5_prc 1.0))) (- 0.0 (* p_crd_crm n5_crm))
                   (- 0.0 (* p_crd_bias 1.0))))))))

       (mi1_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ mi1_ginc (+ n1_prc 1.0)))
                   (* p_crd_wth (/ mi1_wth (+ n1_prc 1.0))) (- 0.0 (* p_crd_crm n1_crm))))))))
       (mi2_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ mi2_ginc (+ n2_prc 1.0)))
                   (* p_crd_wth (/ mi2_wth (+ n2_prc 1.0))) (- 0.0 (* p_crd_crm n2_crm))))))))
       (mi3_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ mi3_ginc (+ n3_prc 1.0)))
                   (* p_crd_wth (/ mi3_wth (+ n3_prc 1.0))) (- 0.0 (* p_crd_crm n3_crm))))))))
       (mi4_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ mi4_ginc (+ n4_prc 1.0)))
                   (* p_crd_wth (/ mi4_wth (+ n4_prc 1.0))) (- 0.0 (* p_crd_crm n4_crm))))))))
       (mi5_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ mi5_ginc (+ n5_prc 1.0)))
                   (* p_crd_wth (/ mi5_wth (+ n5_prc 1.0))) (- 0.0 (* p_crd_crm n5_crm))))))))

       (hi1_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ hi1_ginc (+ n1_prc 1.0)))
                   (* p_crd_wth (/ hi1_wth (+ n1_prc 1.0))) (- 0.0 (* p_crd_crm n1_crm))))))))
       (hi2_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ hi2_ginc (+ n2_prc 1.0)))
                   (* p_crd_wth (/ hi2_wth (+ n2_prc 1.0))) (- 0.0 (* p_crd_crm n2_crm))))))))
       (hi3_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ hi3_ginc (+ n3_prc 1.0)))
                   (* p_crd_wth (/ hi3_wth (+ n3_prc 1.0))) (- 0.0 (* p_crd_crm n3_crm))))))))
       (hi4_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ hi4_ginc (+ n4_prc 1.0)))
                   (* p_crd_wth (/ hi4_wth (+ n4_prc 1.0))) (- 0.0 (* p_crd_crm n4_crm))))))))
       (hi5_cred  (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_crd_inc (/ hi5_ginc (+ n5_prc 1.0)))
                   (* p_crd_wth (/ hi5_wth (+ n5_prc 1.0))) (- 0.0 (* p_crd_crm n5_crm))))))))

       ;; =================================================================
       ;; MODULE 4: Housing Costs
       ;; =================================================================
       ;; Ownership fraction: sigmoid(wealth - down_payment_needed)
       ;; Housing cost = own_frac × mortgage + (1 - own_frac) × rent
       ;; Subsidy reduces effective rent for low income
       (lo1_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- lo1_wth (* p_own_dwn n1_prc))))))))
       (lo1_hcst  (+ (* lo1_own (* p_own_mort n1_prc)) (* (- 1.0 lo1_own) n1_rnt (- 1.0 p_pol_sub))))
       (lo2_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- lo2_wth (* p_own_dwn n2_prc))))))))
       (lo2_hcst  (+ (* lo2_own (* p_own_mort n2_prc)) (* (- 1.0 lo2_own) n2_rnt (- 1.0 p_pol_sub))))
       (lo3_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- lo3_wth (* p_own_dwn n3_prc))))))))
       (lo3_hcst  (+ (* lo3_own (* p_own_mort n3_prc)) (* (- 1.0 lo3_own) n3_rnt (- 1.0 p_pol_sub))))
       (lo4_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- lo4_wth (* p_own_dwn n4_prc))))))))
       (lo4_hcst  (+ (* lo4_own (* p_own_mort n4_prc)) (* (- 1.0 lo4_own) n4_rnt (- 1.0 p_pol_sub))))
       (lo5_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- lo5_wth (* p_own_dwn n5_prc))))))))
       (lo5_hcst  (+ (* lo5_own (* p_own_mort n5_prc)) (* (- 1.0 lo5_own) n5_rnt (- 1.0 p_pol_sub))))

       ;; Mid and high income: no subsidy, same ownership model
       (mi1_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- mi1_wth (* p_own_dwn n1_prc))))))))
       (mi1_hcst  (+ (* mi1_own (* p_own_mort n1_prc)) (* (- 1.0 mi1_own) n1_rnt)))
       (mi2_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- mi2_wth (* p_own_dwn n2_prc))))))))
       (mi2_hcst  (+ (* mi2_own (* p_own_mort n2_prc)) (* (- 1.0 mi2_own) n2_rnt)))
       (mi3_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- mi3_wth (* p_own_dwn n3_prc))))))))
       (mi3_hcst  (+ (* mi3_own (* p_own_mort n3_prc)) (* (- 1.0 mi3_own) n3_rnt)))
       (mi4_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- mi4_wth (* p_own_dwn n4_prc))))))))
       (mi4_hcst  (+ (* mi4_own (* p_own_mort n4_prc)) (* (- 1.0 mi4_own) n4_rnt)))
       (mi5_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- mi5_wth (* p_own_dwn n5_prc))))))))
       (mi5_hcst  (+ (* mi5_own (* p_own_mort n5_prc)) (* (- 1.0 mi5_own) n5_rnt)))

       (hi1_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- hi1_wth (* p_own_dwn n1_prc))))))))
       (hi1_hcst  (+ (* hi1_own (* p_own_mort n1_prc)) (* (- 1.0 hi1_own) n1_rnt)))
       (hi2_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- hi2_wth (* p_own_dwn n2_prc))))))))
       (hi2_hcst  (+ (* hi2_own (* p_own_mort n2_prc)) (* (- 1.0 hi2_own) n2_rnt)))
       (hi3_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- hi3_wth (* p_own_dwn n3_prc))))))))
       (hi3_hcst  (+ (* hi3_own (* p_own_mort n3_prc)) (* (- 1.0 hi3_own) n3_rnt)))
       (hi4_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- hi4_wth (* p_own_dwn n4_prc))))))))
       (hi4_hcst  (+ (* hi4_own (* p_own_mort n4_prc)) (* (- 1.0 hi4_own) n4_rnt)))
       (hi5_own  (/ 1.0 (+ 1.0 (exp (- 0.0 (* p_own_wth (- hi5_wth (* p_own_dwn n5_prc))))))))
       (hi5_hcst  (+ (* hi5_own (* p_own_mort n5_prc)) (* (- 1.0 hi5_own) n5_rnt)))

       ;; =================================================================
       ;; MODULE 5: Wealth Dynamics
       ;; =================================================================
       ;; d_wth = net_income - consumption - housing_cost + wealth_return - property_tax
       ;; net_income = gross × (1 - tax_rate)
       ;; consumption = propensity × gross + subsistence
       (d_lo1_wth  (+ (* lo1_ginc (- 1.0 p_inc_tax p_cns_lo)) (- 0.0 p_cns_base lo1_hcst) (* p_wth_ret lo1_wth) (- 0.0 (* p_ptax n1_prc lo1_own))))
       (d_lo2_wth  (+ (* lo2_ginc (- 1.0 p_inc_tax p_cns_lo)) (- 0.0 p_cns_base lo2_hcst) (* p_wth_ret lo2_wth) (- 0.0 (* p_ptax n2_prc lo2_own))))
       (d_lo3_wth  (+ (* lo3_ginc (- 1.0 p_inc_tax p_cns_lo)) (- 0.0 p_cns_base lo3_hcst) (* p_wth_ret lo3_wth) (- 0.0 (* p_ptax n3_prc lo3_own))))
       (d_lo4_wth  (+ (* lo4_ginc (- 1.0 p_inc_tax p_cns_lo)) (- 0.0 p_cns_base lo4_hcst) (* p_wth_ret lo4_wth) (- 0.0 (* p_ptax n4_prc lo4_own))))
       (d_lo5_wth  (+ (* lo5_ginc (- 1.0 p_inc_tax p_cns_lo)) (- 0.0 p_cns_base lo5_hcst) (* p_wth_ret lo5_wth) (- 0.0 (* p_ptax n5_prc lo5_own))))

       (d_mi1_wth  (+ (* mi1_ginc (- 1.0 p_inc_tax p_cns_mi)) (- 0.0 p_cns_base mi1_hcst) (* p_wth_ret mi1_wth) (- 0.0 (* p_ptax n1_prc mi1_own))))
       (d_mi2_wth  (+ (* mi2_ginc (- 1.0 p_inc_tax p_cns_mi)) (- 0.0 p_cns_base mi2_hcst) (* p_wth_ret mi2_wth) (- 0.0 (* p_ptax n2_prc mi2_own))))
       (d_mi3_wth  (+ (* mi3_ginc (- 1.0 p_inc_tax p_cns_mi)) (- 0.0 p_cns_base mi3_hcst) (* p_wth_ret mi3_wth) (- 0.0 (* p_ptax n3_prc mi3_own))))
       (d_mi4_wth  (+ (* mi4_ginc (- 1.0 p_inc_tax p_cns_mi)) (- 0.0 p_cns_base mi4_hcst) (* p_wth_ret mi4_wth) (- 0.0 (* p_ptax n4_prc mi4_own))))
       (d_mi5_wth  (+ (* mi5_ginc (- 1.0 p_inc_tax p_cns_mi)) (- 0.0 p_cns_base mi5_hcst) (* p_wth_ret mi5_wth) (- 0.0 (* p_ptax n5_prc mi5_own))))

       (d_hi1_wth  (+ (* hi1_ginc (- 1.0 p_inc_tax p_cns_hi)) (- 0.0 p_cns_base hi1_hcst) (* p_wth_ret hi1_wth) (- 0.0 (* p_ptax n1_prc hi1_own))))
       (d_hi2_wth  (+ (* hi2_ginc (- 1.0 p_inc_tax p_cns_hi)) (- 0.0 p_cns_base hi2_hcst) (* p_wth_ret hi2_wth) (- 0.0 (* p_ptax n2_prc hi2_own))))
       (d_hi3_wth  (+ (* hi3_ginc (- 1.0 p_inc_tax p_cns_hi)) (- 0.0 p_cns_base hi3_hcst) (* p_wth_ret hi3_wth) (- 0.0 (* p_ptax n3_prc hi3_own))))
       (d_hi4_wth  (+ (* hi4_ginc (- 1.0 p_inc_tax p_cns_hi)) (- 0.0 p_cns_base hi4_hcst) (* p_wth_ret hi4_wth) (- 0.0 (* p_ptax n4_prc hi4_own))))
       (d_hi5_wth  (+ (* hi5_ginc (- 1.0 p_inc_tax p_cns_hi)) (- 0.0 p_cns_base hi5_hcst) (* p_wth_ret hi5_wth) (- 0.0 (* p_ptax n5_prc hi5_own))))

       ;; =================================================================
       ;; MODULE 6: Housing Market
       ;; =================================================================
       ;; Demand: population × credit × sigmoid(preference)
       ;; Bounded [0, pop] to prevent exponential overflow in long runs
       ;; Preference: school + amenity - crime - affordability + speculation
       (dem_1  (+ (* lo1_pop lo1_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n1_sch) (* p_prf_amn n1_amn)
                    (- 0.0 (* p_prf_crm n1_crm)) (- 0.0 (* p_prf_prc (/ n1_prc (+ lo1_ginc 1.0))))))))))
                  (* mi1_pop mi1_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n1_sch) (* p_prf_amn n1_amn)
                    (- 0.0 (* p_prf_crm n1_crm)) (- 0.0 (* p_prf_prc (/ n1_prc (+ mi1_ginc 1.0))))))))))
                  (* hi1_pop hi1_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n1_sch) (* p_prf_amn n1_amn)
                    (- 0.0 (* p_prf_crm n1_crm)) (- 0.0 (* p_prf_prc (/ n1_prc (+ hi1_ginc 1.0))))
                    (* p_hsg_spec hi1_wth 0.001)))))))))
       (dem_2  (+ (* lo2_pop lo2_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n2_sch) (* p_prf_amn n2_amn)
                    (- 0.0 (* p_prf_crm n2_crm)) (- 0.0 (* p_prf_prc (/ n2_prc (+ lo2_ginc 1.0))))))))))
                  (* mi2_pop mi2_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n2_sch) (* p_prf_amn n2_amn)
                    (- 0.0 (* p_prf_crm n2_crm)) (- 0.0 (* p_prf_prc (/ n2_prc (+ mi2_ginc 1.0))))))))))
                  (* hi2_pop hi2_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n2_sch) (* p_prf_amn n2_amn)
                    (- 0.0 (* p_prf_crm n2_crm)) (- 0.0 (* p_prf_prc (/ n2_prc (+ hi2_ginc 1.0))))
                    (* p_hsg_spec hi2_wth 0.001)))))))))
       (dem_3  (+ (* lo3_pop lo3_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n3_sch) (* p_prf_amn n3_amn)
                    (- 0.0 (* p_prf_crm n3_crm)) (- 0.0 (* p_prf_prc (/ n3_prc (+ lo3_ginc 1.0))))))))))
                  (* mi3_pop mi3_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n3_sch) (* p_prf_amn n3_amn)
                    (- 0.0 (* p_prf_crm n3_crm)) (- 0.0 (* p_prf_prc (/ n3_prc (+ mi3_ginc 1.0))))))))))
                  (* hi3_pop hi3_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n3_sch) (* p_prf_amn n3_amn)
                    (- 0.0 (* p_prf_crm n3_crm)) (- 0.0 (* p_prf_prc (/ n3_prc (+ hi3_ginc 1.0))))
                    (* p_hsg_spec hi3_wth 0.001)))))))))
       (dem_4  (+ (* lo4_pop lo4_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n4_sch) (* p_prf_amn n4_amn)
                    (- 0.0 (* p_prf_crm n4_crm)) (- 0.0 (* p_prf_prc (/ n4_prc (+ lo4_ginc 1.0))))))))))
                  (* mi4_pop mi4_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n4_sch) (* p_prf_amn n4_amn)
                    (- 0.0 (* p_prf_crm n4_crm)) (- 0.0 (* p_prf_prc (/ n4_prc (+ mi4_ginc 1.0))))))))))
                  (* hi4_pop hi4_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n4_sch) (* p_prf_amn n4_amn)
                    (- 0.0 (* p_prf_crm n4_crm)) (- 0.0 (* p_prf_prc (/ n4_prc (+ hi4_ginc 1.0))))
                    (* p_hsg_spec hi4_wth 0.001)))))))))
       (dem_5  (+ (* lo5_pop lo5_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n5_sch) (* p_prf_amn n5_amn)
                    (- 0.0 (* p_prf_crm n5_crm)) (- 0.0 (* p_prf_prc (/ n5_prc (+ lo5_ginc 1.0))))))))))
                  (* mi5_pop mi5_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n5_sch) (* p_prf_amn n5_amn)
                    (- 0.0 (* p_prf_crm n5_crm)) (- 0.0 (* p_prf_prc (/ n5_prc (+ mi5_ginc 1.0))))))))))
                  (* hi5_pop hi5_cred (/ 1.0 (+ 1.0 (exp (- 0.0 (+ (* p_prf_sch n5_sch) (* p_prf_amn n5_amn)
                    (- 0.0 (* p_prf_crm n5_crm)) (- 0.0 (* p_prf_prc (/ n5_prc (+ hi5_ginc 1.0))))
                    (* p_hsg_spec hi5_wth 0.001)))))))))

       ;; Supply: capacity × elasticity + policy
       (sup_1  (+ (* cap_1 (+ 1.0 (* p_hsg_sel (/ n1_prc avg_prc)))) (* p_pol_zone e_hsup)))
       (sup_2  (+ (* cap_2 (+ 1.0 (* p_hsg_sel (/ n2_prc avg_prc)))) (* p_pol_zone e_hsup)))
       (sup_3  (+ (* cap_3 (+ 1.0 (* p_hsg_sel (/ n3_prc avg_prc)))) (* p_pol_zone e_hsup)))
       (sup_4  (+ (* cap_4 (+ 1.0 (* p_hsg_sel (/ n4_prc avg_prc)))) (* p_pol_zone e_hsup)))
       (sup_5  (+ (* cap_5 (+ 1.0 (* p_hsg_sel (/ n5_prc avg_prc)))) (* p_pol_zone e_hsup)))

       ;; Price dynamics: demand/supply + momentum + mean reversion - depreciation
       (d_n1_prc  (* p_hsg_padj (+ (* n1_prc (- (/ dem_1 (+ sup_1 0.1)) 1.0))
                    (* p_hsg_mom (- n1_prc avg_prc)) (* p_hsg_mrev (- avg_prc n1_prc))
                    (- 0.0 (* p_hsg_dep n1_prc)))))
       (d_n2_prc  (* p_hsg_padj (+ (* n2_prc (- (/ dem_2 (+ sup_2 0.1)) 1.0))
                    (* p_hsg_mom (- n2_prc avg_prc)) (* p_hsg_mrev (- avg_prc n2_prc))
                    (- 0.0 (* p_hsg_dep n2_prc)))))
       (d_n3_prc  (* p_hsg_padj (+ (* n3_prc (- (/ dem_3 (+ sup_3 0.1)) 1.0))
                    (* p_hsg_mom (- n3_prc avg_prc)) (* p_hsg_mrev (- avg_prc n3_prc))
                    (- 0.0 (* p_hsg_dep n3_prc)))))
       (d_n4_prc  (* p_hsg_padj (+ (* n4_prc (- (/ dem_4 (+ sup_4 0.1)) 1.0))
                    (* p_hsg_mom (- n4_prc avg_prc)) (* p_hsg_mrev (- avg_prc n4_prc))
                    (- 0.0 (* p_hsg_dep n4_prc)))))
       (d_n5_prc  (* p_hsg_padj (+ (* n5_prc (- (/ dem_5 (+ sup_5 0.1)) 1.0))
                    (* p_hsg_mom (- n5_prc avg_prc)) (* p_hsg_mrev (- avg_prc n5_prc))
                    (- 0.0 (* p_hsg_dep n5_prc)))))

       ;; Rent dynamics: adjust toward yield target + rent control
       (d_n1_rnt  (* p_hsg_radj (- (* p_hsg_ryld n1_prc) n1_rnt (* p_pol_rent n1_rnt))))
       (d_n2_rnt  (* p_hsg_radj (- (* p_hsg_ryld n2_prc) n2_rnt (* p_pol_rent n2_rnt))))
       (d_n3_rnt  (* p_hsg_radj (- (* p_hsg_ryld n3_prc) n3_rnt (* p_pol_rent n3_rnt))))
       (d_n4_rnt  (* p_hsg_radj (- (* p_hsg_ryld n4_prc) n4_rnt (* p_pol_rent n4_rnt))))
       (d_n5_rnt  (* p_hsg_radj (- (* p_hsg_ryld n5_prc) n5_rnt (* p_pol_rent n5_rnt))))

       ;; =================================================================
       ;; MODULE 7: Schools & Human Capital
       ;; =================================================================
       ;; School quality: funding (from local income tax) + peer effects - decay
       ;; With equalization policy and spillover from neighbors
       (fund_1  (+ (* p_sch_tax avginc_1) p_sch_base (* p_pol_sch (- avg_prc avginc_1))))
       (fund_2  (+ (* p_sch_tax avginc_2) p_sch_base (* p_pol_sch (- avg_prc avginc_2))))
       (fund_3  (+ (* p_sch_tax avginc_3) p_sch_base (* p_pol_sch (- avg_prc avginc_3))))
       (fund_4  (+ (* p_sch_tax avginc_4) p_sch_base (* p_pol_sch (- avg_prc avginc_4))))
       (fund_5  (+ (* p_sch_tax avginc_5) p_sch_base (* p_pol_sch (- avg_prc avginc_5))))

       ;; Weighted skill (peer quality) per neighborhood
       (peer_1  (/ (+ (* lo1_pop lo1_skl) (* mi1_pop mi1_skl) (* hi1_pop hi1_skl)) (+ tot_1 0.1)))
       (peer_2  (/ (+ (* lo2_pop lo2_skl) (* mi2_pop mi2_skl) (* hi2_pop hi2_skl)) (+ tot_2 0.1)))
       (peer_3  (/ (+ (* lo3_pop lo3_skl) (* mi3_pop mi3_skl) (* hi3_pop hi3_skl)) (+ tot_3 0.1)))
       (peer_4  (/ (+ (* lo4_pop lo4_skl) (* mi4_pop mi4_skl) (* hi4_pop hi4_skl)) (+ tot_4 0.1)))
       (peer_5  (/ (+ (* lo5_pop lo5_skl) (* mi5_pop mi5_skl) (* hi5_pop hi5_skl)) (+ tot_5 0.1)))

       ;; School quality change with neighbor spillover
       (d_n1_sch  (+ (* p_sch_fund fund_1) (* p_sch_peer peer_1) (- 0.0 (* p_sch_dec n1_sch))
                     (* p_sch_spill (- (/ (+ n2_sch n3_sch) 2.0) n1_sch))))
       (d_n2_sch  (+ (* p_sch_fund fund_2) (* p_sch_peer peer_2) (- 0.0 (* p_sch_dec n2_sch))
                     (* p_sch_spill (- (/ (+ n1_sch n3_sch n4_sch) 3.0) n2_sch))))
       (d_n3_sch  (+ (* p_sch_fund fund_3) (* p_sch_peer peer_3) (- 0.0 (* p_sch_dec n3_sch))
                     (* p_sch_spill (- (/ (+ n1_sch n2_sch n5_sch) 3.0) n3_sch))))
       (d_n4_sch  (+ (* p_sch_fund fund_4) (* p_sch_peer peer_4) (- 0.0 (* p_sch_dec n4_sch))
                     (* p_sch_spill (- (/ (+ n2_sch n5_sch) 2.0) n4_sch))))
       (d_n5_sch  (+ (* p_sch_fund fund_5) (* p_sch_peer peer_5) (- 0.0 (* p_sch_dec n5_sch))
                     (* p_sch_spill (- (/ (+ n3_sch n4_sch) 2.0) n5_sch))))

       ;; Skill formation: school + peer + parental income - decay
       (d_lo1_skl  (+ (* p_skl_sch n1_sch) (* p_skl_peer peer_1) (* p_skl_inc (log (+ lo1_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec lo1_skl))))
       (d_lo2_skl  (+ (* p_skl_sch n2_sch) (* p_skl_peer peer_2) (* p_skl_inc (log (+ lo2_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec lo2_skl))))
       (d_lo3_skl  (+ (* p_skl_sch n3_sch) (* p_skl_peer peer_3) (* p_skl_inc (log (+ lo3_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec lo3_skl))))
       (d_lo4_skl  (+ (* p_skl_sch n4_sch) (* p_skl_peer peer_4) (* p_skl_inc (log (+ lo4_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec lo4_skl))))
       (d_lo5_skl  (+ (* p_skl_sch n5_sch) (* p_skl_peer peer_5) (* p_skl_inc (log (+ lo5_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec lo5_skl))))

       (d_mi1_skl  (+ (* p_skl_sch n1_sch) (* p_skl_peer peer_1) (* p_skl_inc (log (+ mi1_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec mi1_skl))))
       (d_mi2_skl  (+ (* p_skl_sch n2_sch) (* p_skl_peer peer_2) (* p_skl_inc (log (+ mi2_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec mi2_skl))))
       (d_mi3_skl  (+ (* p_skl_sch n3_sch) (* p_skl_peer peer_3) (* p_skl_inc (log (+ mi3_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec mi3_skl))))
       (d_mi4_skl  (+ (* p_skl_sch n4_sch) (* p_skl_peer peer_4) (* p_skl_inc (log (+ mi4_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec mi4_skl))))
       (d_mi5_skl  (+ (* p_skl_sch n5_sch) (* p_skl_peer peer_5) (* p_skl_inc (log (+ mi5_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec mi5_skl))))

       (d_hi1_skl  (+ (* p_skl_sch n1_sch) (* p_skl_peer peer_1) (* p_skl_inc (log (+ hi1_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec hi1_skl))))
       (d_hi2_skl  (+ (* p_skl_sch n2_sch) (* p_skl_peer peer_2) (* p_skl_inc (log (+ hi2_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec hi2_skl))))
       (d_hi3_skl  (+ (* p_skl_sch n3_sch) (* p_skl_peer peer_3) (* p_skl_inc (log (+ hi3_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec hi3_skl))))
       (d_hi4_skl  (+ (* p_skl_sch n4_sch) (* p_skl_peer peer_4) (* p_skl_inc (log (+ hi4_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec hi4_skl))))
       (d_hi5_skl  (+ (* p_skl_sch n5_sch) (* p_skl_peer peer_5) (* p_skl_inc (log (+ hi5_ginc 1.0))) p_skl_base (- 0.0 (* p_skl_dec hi5_skl))))

       ;; =================================================================
       ;; MODULE 8: Migration (population-conserving)
       ;; =================================================================
       ;; Attractiveness = school + amenity - crime - price/income + job access
       ;; Low income
       (attr_lo_1  (+ (* p_att_sch n1_sch) (* p_att_amn n1_amn) (- 0.0 (* p_att_crm n1_crm))
                      (- 0.0 (* p_att_prc (/ n1_prc (+ avginc_lo 1.0)))) (* p_att_job jacc_1) p_mig_att))
       (attr_lo_2  (+ (* p_att_sch n2_sch) (* p_att_amn n2_amn) (- 0.0 (* p_att_crm n2_crm))
                      (- 0.0 (* p_att_prc (/ n2_prc (+ avginc_lo 1.0)))) (* p_att_job jacc_2) p_mig_att))
       (attr_lo_3  (+ (* p_att_sch n3_sch) (* p_att_amn n3_amn) (- 0.0 (* p_att_crm n3_crm))
                      (- 0.0 (* p_att_prc (/ n3_prc (+ avginc_lo 1.0)))) (* p_att_job jacc_3) p_mig_att))
       (attr_lo_4  (+ (* p_att_sch n4_sch) (* p_att_amn n4_amn) (- 0.0 (* p_att_crm n4_crm))
                      (- 0.0 (* p_att_prc (/ n4_prc (+ avginc_lo 1.0)))) (* p_att_job jacc_4) p_mig_att))
       (attr_lo_5  (+ (* p_att_sch n5_sch) (* p_att_amn n5_amn) (- 0.0 (* p_att_crm n5_crm))
                      (- 0.0 (* p_att_prc (/ n5_prc (+ avginc_lo 1.0)))) (* p_att_job jacc_5) p_mig_att))
       ;; Population-weighted average attractiveness (ensures conservation)
       (avgat_lo  (/ (+ (* lo1_pop attr_lo_1) (* lo2_pop attr_lo_2) (* lo3_pop attr_lo_3)
                        (* lo4_pop attr_lo_4) (* lo5_pop attr_lo_5)) (+ tot_lo 0.01)))
       ;; Migration flow: move toward more attractive, conserves population
       (d_lo1_pop  (+ (* p_mig_lo lo1_pop (- (/ attr_lo_1 (+ avgat_lo 0.01)) 1.0)) (* p_mac_pop lo1_pop e_imm)))
       (d_lo2_pop  (+ (* p_mig_lo lo2_pop (- (/ attr_lo_2 (+ avgat_lo 0.01)) 1.0)) (* p_mac_pop lo2_pop e_imm)))
       (d_lo3_pop  (+ (* p_mig_lo lo3_pop (- (/ attr_lo_3 (+ avgat_lo 0.01)) 1.0)) (* p_mac_pop lo3_pop e_imm)))
       (d_lo4_pop  (+ (* p_mig_lo lo4_pop (- (/ attr_lo_4 (+ avgat_lo 0.01)) 1.0)) (* p_mac_pop lo4_pop e_imm)))
       (d_lo5_pop  (+ (* p_mig_lo lo5_pop (- (/ attr_lo_5 (+ avgat_lo 0.01)) 1.0)) (* p_mac_pop lo5_pop e_imm)))

       ;; Mid income
       (attr_mi_1  (+ (* p_att_sch n1_sch) (* p_att_amn n1_amn) (- 0.0 (* p_att_crm n1_crm))
                      (- 0.0 (* p_att_prc (/ n1_prc (+ avginc_mi 1.0)))) (* p_att_job jacc_1) p_mig_att))
       (attr_mi_2  (+ (* p_att_sch n2_sch) (* p_att_amn n2_amn) (- 0.0 (* p_att_crm n2_crm))
                      (- 0.0 (* p_att_prc (/ n2_prc (+ avginc_mi 1.0)))) (* p_att_job jacc_2) p_mig_att))
       (attr_mi_3  (+ (* p_att_sch n3_sch) (* p_att_amn n3_amn) (- 0.0 (* p_att_crm n3_crm))
                      (- 0.0 (* p_att_prc (/ n3_prc (+ avginc_mi 1.0)))) (* p_att_job jacc_3) p_mig_att))
       (attr_mi_4  (+ (* p_att_sch n4_sch) (* p_att_amn n4_amn) (- 0.0 (* p_att_crm n4_crm))
                      (- 0.0 (* p_att_prc (/ n4_prc (+ avginc_mi 1.0)))) (* p_att_job jacc_4) p_mig_att))
       (attr_mi_5  (+ (* p_att_sch n5_sch) (* p_att_amn n5_amn) (- 0.0 (* p_att_crm n5_crm))
                      (- 0.0 (* p_att_prc (/ n5_prc (+ avginc_mi 1.0)))) (* p_att_job jacc_5) p_mig_att))
       (avgat_mi  (/ (+ (* mi1_pop attr_mi_1) (* mi2_pop attr_mi_2) (* mi3_pop attr_mi_3)
                        (* mi4_pop attr_mi_4) (* mi5_pop attr_mi_5)) (+ tot_mi 0.01)))
       (d_mi1_pop  (+ (* p_mig_mi mi1_pop (- (/ attr_mi_1 (+ avgat_mi 0.01)) 1.0)) (* p_mac_pop mi1_pop e_imm)))
       (d_mi2_pop  (+ (* p_mig_mi mi2_pop (- (/ attr_mi_2 (+ avgat_mi 0.01)) 1.0)) (* p_mac_pop mi2_pop e_imm)))
       (d_mi3_pop  (+ (* p_mig_mi mi3_pop (- (/ attr_mi_3 (+ avgat_mi 0.01)) 1.0)) (* p_mac_pop mi3_pop e_imm)))
       (d_mi4_pop  (+ (* p_mig_mi mi4_pop (- (/ attr_mi_4 (+ avgat_mi 0.01)) 1.0)) (* p_mac_pop mi4_pop e_imm)))
       (d_mi5_pop  (+ (* p_mig_mi mi5_pop (- (/ attr_mi_5 (+ avgat_mi 0.01)) 1.0)) (* p_mac_pop mi5_pop e_imm)))

       ;; High income
       (attr_hi_1  (+ (* p_att_sch n1_sch) (* p_att_amn n1_amn) (- 0.0 (* p_att_crm n1_crm))
                      (- 0.0 (* p_att_prc (/ n1_prc (+ avginc_hi 1.0)))) (* p_att_job jacc_1) p_mig_att))
       (attr_hi_2  (+ (* p_att_sch n2_sch) (* p_att_amn n2_amn) (- 0.0 (* p_att_crm n2_crm))
                      (- 0.0 (* p_att_prc (/ n2_prc (+ avginc_hi 1.0)))) (* p_att_job jacc_2) p_mig_att))
       (attr_hi_3  (+ (* p_att_sch n3_sch) (* p_att_amn n3_amn) (- 0.0 (* p_att_crm n3_crm))
                      (- 0.0 (* p_att_prc (/ n3_prc (+ avginc_hi 1.0)))) (* p_att_job jacc_3) p_mig_att))
       (attr_hi_4  (+ (* p_att_sch n4_sch) (* p_att_amn n4_amn) (- 0.0 (* p_att_crm n4_crm))
                      (- 0.0 (* p_att_prc (/ n4_prc (+ avginc_hi 1.0)))) (* p_att_job jacc_4) p_mig_att))
       (attr_hi_5  (+ (* p_att_sch n5_sch) (* p_att_amn n5_amn) (- 0.0 (* p_att_crm n5_crm))
                      (- 0.0 (* p_att_prc (/ n5_prc (+ avginc_hi 1.0)))) (* p_att_job jacc_5) p_mig_att))
       (avgat_hi  (/ (+ (* hi1_pop attr_hi_1) (* hi2_pop attr_hi_2) (* hi3_pop attr_hi_3)
                        (* hi4_pop attr_hi_4) (* hi5_pop attr_hi_5)) (+ tot_hi 0.01)))
       (d_hi1_pop  (+ (* p_mig_hi hi1_pop (- (/ attr_hi_1 (+ avgat_hi 0.01)) 1.0)) (* p_mac_pop hi1_pop e_imm)))
       (d_hi2_pop  (+ (* p_mig_hi hi2_pop (- (/ attr_hi_2 (+ avgat_hi 0.01)) 1.0)) (* p_mac_pop hi2_pop e_imm)))
       (d_hi3_pop  (+ (* p_mig_hi hi3_pop (- (/ attr_hi_3 (+ avgat_hi 0.01)) 1.0)) (* p_mac_pop hi3_pop e_imm)))
       (d_hi4_pop  (+ (* p_mig_hi hi4_pop (- (/ attr_hi_4 (+ avgat_hi 0.01)) 1.0)) (* p_mac_pop hi4_pop e_imm)))
       (d_hi5_pop  (+ (* p_mig_hi hi5_pop (- (/ attr_hi_5 (+ avgat_hi 0.01)) 1.0)) (* p_mac_pop hi5_pop e_imm)))

       ;; =================================================================
       ;; MODULE 9: Neighborhood Dynamics
       ;; =================================================================
       ;; Amenity: investment from income + public investment - decay - crime damage
       ;; With spillover from adjacent neighborhoods
       (d_n1_amn  (+ (* p_nbr_ainc avginc_1 0.001) (* p_nbr_ainv e_inv) (- 0.0 (* p_nbr_adec n1_amn))
                     (- 0.0 (* p_nbr_acrm n1_crm))
                     (* p_nbr_spamn (- (/ (+ n2_amn n3_amn) 2.0) n1_amn))))
       (d_n2_amn  (+ (* p_nbr_ainc avginc_2 0.001) (* p_nbr_ainv e_inv) (- 0.0 (* p_nbr_adec n2_amn))
                     (- 0.0 (* p_nbr_acrm n2_crm))
                     (* p_nbr_spamn (- (/ (+ n1_amn n3_amn n4_amn) 3.0) n2_amn))))
       (d_n3_amn  (+ (* p_nbr_ainc avginc_3 0.001) (* p_nbr_ainv e_inv) (- 0.0 (* p_nbr_adec n3_amn))
                     (- 0.0 (* p_nbr_acrm n3_crm))
                     (* p_nbr_spamn (- (/ (+ n1_amn n2_amn n5_amn) 3.0) n3_amn))))
       (d_n4_amn  (+ (* p_nbr_ainc avginc_4 0.001) (* p_nbr_ainv e_inv) (- 0.0 (* p_nbr_adec n4_amn))
                     (- 0.0 (* p_nbr_acrm n4_crm))
                     (* p_nbr_spamn (- (/ (+ n2_amn n5_amn) 2.0) n4_amn))))
       (d_n5_amn  (+ (* p_nbr_ainc avginc_5 0.001) (* p_nbr_ainv e_inv) (- 0.0 (* p_nbr_adec n5_amn))
                     (- 0.0 (* p_nbr_acrm n5_crm))
                     (* p_nbr_spamn (- (/ (+ n3_amn n4_amn) 2.0) n5_amn))))

       ;; Crime: baseline - income effect - policing + segregation + spillover - decay
       ;; Segregation: deviation of income mix from city average increases crime
       (seg_1  (/ (+ (- avginc_1 avginc_lo) (- avginc_1 avginc_hi)) (+ avginc_1 1.0)))
       (seg_2  (/ (+ (- avginc_2 avginc_lo) (- avginc_2 avginc_hi)) (+ avginc_2 1.0)))
       (seg_3  (/ (+ (- avginc_3 avginc_lo) (- avginc_3 avginc_hi)) (+ avginc_3 1.0)))
       (seg_4  (/ (+ (- avginc_4 avginc_lo) (- avginc_4 avginc_hi)) (+ avginc_4 1.0)))
       (seg_5  (/ (+ (- avginc_5 avginc_lo) (- avginc_5 avginc_hi)) (+ avginc_5 1.0)))

       (d_n1_crm  (+ p_nbr_cbase (- 0.0 (* p_nbr_cinc avginc_1 0.001)) (- 0.0 (* p_nbr_cpol e_inv))
                     (* p_soc_seg seg_1) (- 0.0 (* p_nbr_cdec n1_crm))
                     (* p_nbr_spcrm (- (/ (+ n2_crm n3_crm) 2.0) n1_crm))))
       (d_n2_crm  (+ p_nbr_cbase (- 0.0 (* p_nbr_cinc avginc_2 0.001)) (- 0.0 (* p_nbr_cpol e_inv))
                     (* p_soc_seg seg_2) (- 0.0 (* p_nbr_cdec n2_crm))
                     (* p_nbr_spcrm (- (/ (+ n1_crm n3_crm n4_crm) 3.0) n2_crm))))
       (d_n3_crm  (+ p_nbr_cbase (- 0.0 (* p_nbr_cinc avginc_3 0.001)) (- 0.0 (* p_nbr_cpol e_inv))
                     (* p_soc_seg seg_3) (- 0.0 (* p_nbr_cdec n3_crm))
                     (* p_nbr_spcrm (- (/ (+ n1_crm n2_crm n5_crm) 3.0) n3_crm))))
       (d_n4_crm  (+ p_nbr_cbase (- 0.0 (* p_nbr_cinc avginc_4 0.001)) (- 0.0 (* p_nbr_cpol e_inv))
                     (* p_soc_seg seg_4) (- 0.0 (* p_nbr_cdec n4_crm))
                     (* p_nbr_spcrm (- (/ (+ n2_crm n5_crm) 2.0) n4_crm))))
       (d_n5_crm  (+ p_nbr_cbase (- 0.0 (* p_nbr_cinc avginc_5 0.001)) (- 0.0 (* p_nbr_cpol e_inv))
                     (* p_soc_seg seg_5) (- 0.0 (* p_nbr_cdec n5_crm))
                     (* p_nbr_spcrm (- (/ (+ n3_crm n4_crm) 2.0) n5_crm))))

       ;; =================================================================
       ;; MODULE 10: Macro & Banking
       ;; =================================================================
       ;; Total income and wealth
       (total_inc  (+ (* lo1_pop lo1_ginc) (* lo2_pop lo2_ginc) (* lo3_pop lo3_ginc)
                      (* lo4_pop lo4_ginc) (* lo5_pop lo5_ginc)
                      (* mi1_pop mi1_ginc) (* mi2_pop mi2_ginc) (* mi3_pop mi3_ginc)
                      (* mi4_pop mi4_ginc) (* mi5_pop mi5_ginc)
                      (* hi1_pop hi1_ginc) (* hi2_pop hi2_ginc) (* hi3_pop hi3_ginc)
                      (* hi4_pop hi4_ginc) (* hi5_pop hi5_ginc)))

       ;; GDP: labor income + wealth returns + growth
       (gdp_new  (+ gdp (* p_mac_gdp gdp) (* p_mac_lab total_inc 0.01) (* e_grow gdp 0.01)))

       ;; Inflation: demand pressure + housing price effect
       (infl_new  (+ infl (* p_mac_inf (- 0.02 infl))
                     (* p_mac_ihsg (/ (+ d_n1_prc d_n2_prc d_n3_prc d_n4_prc d_n5_prc) (+ avg_prc 1.0)))
                     (* p_mac_idem (- (/ total_inc (+ gdp 1.0)) 1.0))))

       ;; Inequality: ratio of high to low average income
       (ineq_new  (* p_mac_ineq (/ (+ avginc_hi 1.0) (+ avginc_lo 1.0))))

       ;; Credit supply: adjusts with economy
       (crd_new  (+ crd (* p_crd_adj (- gdp_new gdp)) (* p_pol_cred crd 0.01) (- 0.0 (* p_crd_rsk irate crd))))

       ;; Interest rate: base policy rate + risk premium from defaults
       ;; Default pressure from low-income over-leveraged households
       (def_pressure  (+ (* (- 1.0 lo1_emp) lo1_pop) (* (- 1.0 lo2_emp) lo2_pop)
                         (* (- 1.0 lo3_emp) lo3_pop) (* (- 1.0 lo4_emp) lo4_pop)
                         (* (- 1.0 lo5_emp) lo5_pop)))
       (irate_new  (+ e_irate (* p_crd_def (/ def_pressure (+ tot_lo 0.1))))))

      ;; =================================================================
      ;; RECUR: update all 98 state variables
      ;; =================================================================
      (recur
        (- n 1)
        (+ year 1.0)

        ;; --- low income households ---
        (+ lo1_pop d_lo1_pop)
        (+ (* (- 1.0 p_inc_adj) lo1_inc) (* p_inc_adj lo1_ginc))
        (+ lo1_wth d_lo1_wth)
        (+ lo1_skl d_lo1_skl)
        (+ lo2_pop d_lo2_pop)
        (+ (* (- 1.0 p_inc_adj) lo2_inc) (* p_inc_adj lo2_ginc))
        (+ lo2_wth d_lo2_wth)
        (+ lo2_skl d_lo2_skl)
        (+ lo3_pop d_lo3_pop)
        (+ (* (- 1.0 p_inc_adj) lo3_inc) (* p_inc_adj lo3_ginc))
        (+ lo3_wth d_lo3_wth)
        (+ lo3_skl d_lo3_skl)
        (+ lo4_pop d_lo4_pop)
        (+ (* (- 1.0 p_inc_adj) lo4_inc) (* p_inc_adj lo4_ginc))
        (+ lo4_wth d_lo4_wth)
        (+ lo4_skl d_lo4_skl)
        (+ lo5_pop d_lo5_pop)
        (+ (* (- 1.0 p_inc_adj) lo5_inc) (* p_inc_adj lo5_ginc))
        (+ lo5_wth d_lo5_wth)
        (+ lo5_skl d_lo5_skl)

        ;; --- mid income households ---
        (+ mi1_pop d_mi1_pop)
        (+ (* (- 1.0 p_inc_adj) mi1_inc) (* p_inc_adj mi1_ginc))
        (+ mi1_wth d_mi1_wth)
        (+ mi1_skl d_mi1_skl)
        (+ mi2_pop d_mi2_pop)
        (+ (* (- 1.0 p_inc_adj) mi2_inc) (* p_inc_adj mi2_ginc))
        (+ mi2_wth d_mi2_wth)
        (+ mi2_skl d_mi2_skl)
        (+ mi3_pop d_mi3_pop)
        (+ (* (- 1.0 p_inc_adj) mi3_inc) (* p_inc_adj mi3_ginc))
        (+ mi3_wth d_mi3_wth)
        (+ mi3_skl d_mi3_skl)
        (+ mi4_pop d_mi4_pop)
        (+ (* (- 1.0 p_inc_adj) mi4_inc) (* p_inc_adj mi4_ginc))
        (+ mi4_wth d_mi4_wth)
        (+ mi4_skl d_mi4_skl)
        (+ mi5_pop d_mi5_pop)
        (+ (* (- 1.0 p_inc_adj) mi5_inc) (* p_inc_adj mi5_ginc))
        (+ mi5_wth d_mi5_wth)
        (+ mi5_skl d_mi5_skl)

        ;; --- high income households ---
        (+ hi1_pop d_hi1_pop)
        (+ (* (- 1.0 p_inc_adj) hi1_inc) (* p_inc_adj hi1_ginc))
        (+ hi1_wth d_hi1_wth)
        (+ hi1_skl d_hi1_skl)
        (+ hi2_pop d_hi2_pop)
        (+ (* (- 1.0 p_inc_adj) hi2_inc) (* p_inc_adj hi2_ginc))
        (+ hi2_wth d_hi2_wth)
        (+ hi2_skl d_hi2_skl)
        (+ hi3_pop d_hi3_pop)
        (+ (* (- 1.0 p_inc_adj) hi3_inc) (* p_inc_adj hi3_ginc))
        (+ hi3_wth d_hi3_wth)
        (+ hi3_skl d_hi3_skl)
        (+ hi4_pop d_hi4_pop)
        (+ (* (- 1.0 p_inc_adj) hi4_inc) (* p_inc_adj hi4_ginc))
        (+ hi4_wth d_hi4_wth)
        (+ hi4_skl d_hi4_skl)
        (+ hi5_pop d_hi5_pop)
        (+ (* (- 1.0 p_inc_adj) hi5_inc) (* p_inc_adj hi5_ginc))
        (+ hi5_wth d_hi5_wth)
        (+ hi5_skl d_hi5_skl)

        ;; --- neighborhoods ---
        (+ n1_prc d_n1_prc) (+ n1_rnt d_n1_rnt) (+ n1_sch d_n1_sch) (+ n1_amn d_n1_amn) (+ n1_crm d_n1_crm)
        (+ n2_prc d_n2_prc) (+ n2_rnt d_n2_rnt) (+ n2_sch d_n2_sch) (+ n2_amn d_n2_amn) (+ n2_crm d_n2_crm)
        (+ n3_prc d_n3_prc) (+ n3_rnt d_n3_rnt) (+ n3_sch d_n3_sch) (+ n3_amn d_n3_amn) (+ n3_crm d_n3_crm)
        (+ n4_prc d_n4_prc) (+ n4_rnt d_n4_rnt) (+ n4_sch d_n4_sch) (+ n4_amn d_n4_amn) (+ n4_crm d_n4_crm)
        (+ n5_prc d_n5_prc) (+ n5_rnt d_n5_rnt) (+ n5_sch d_n5_sch) (+ n5_amn d_n5_amn) (+ n5_crm d_n5_crm)

        ;; --- labor market ---
        (+ s1_dem d_s1_dem) (+ s1_wge d_s1_wge)
        (+ s2_dem d_s2_dem) (+ s2_wge d_s2_wge)
        (+ s3_dem d_s3_dem) (+ s3_wge d_s3_wge)

        ;; --- banking ---
        crd_new irate_new

        ;; --- macro ---
        gdp_new infl_new ineq_new))))
