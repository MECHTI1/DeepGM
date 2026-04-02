Two projects- 
1) try to distinguish metal type by both predicted coordination sphere and secondaty sphere to third
2) try to understand type of reaction or mechanism by it. EC. 



- ESMC embedding 
- GVP 
- late fusion of GVP and ESMC embedding for good order.
- Add supervised contrastive loss alongside cross-entropy?
- EC classifier or/and metal Atom type classifier


















------------------------------------------------------------------------------------
explanation :
    1)add net_ligand_vector, which are the sum of vectors foemed bewtene the metal ligans to the metal, and
    2) calculate (directly or not) the angle created by the intersection of this vector to the one that which formed between the metal ion to the residue
 For each metal site, compute
     v_net = Σ_i (r_ligand_i - r_metal)
 and
    ||v_net||
 For each nearby residue, compute
     v_res = r_residue - r_metal
     ||v_res||
 and
     cos(θ) = (v_net · v_res) / (||v_net|| ||v_res|| + eps)
  Add to GVP:
  vector features: v_net, v_res
  scalar features: ||v_net||, ||v_res||, cos(θ)
---------------------------------------------------------------------------------------

1) predict with mahomes2 the pdb proteins- like this can have both features and predictions of which site enzymatic and which not.
2) then, create terms: if the metal found within chain that belong to specific EC and it predicted to be enzymatic ->This metal row will be assigned with specific EC Number,
    Otherwise, will not assigned with any EC.
3) keep only the rows with the predicted EC-> they will be the training/testset
