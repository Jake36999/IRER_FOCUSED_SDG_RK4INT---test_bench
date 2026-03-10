CUPY VERSION FAULTY

-The Cupy omega gravity function and rk4 solver application of the conformal carry touple is incorrect-- [kills sim atm! pending fix.]
-IRER_v14 contains decoupled modularised physics suite, this version is much simpler, but stable and able to apply quantum fluid tensor correctly and measure induced pressure solving for omega. 
-test bench - current design focuses on JAX sim deployment and monitoring as well as architectural validaiton via ast walk. --[pending mathematical validation tool via AST walk analysis process ensuring lower debug times. ]


--current development goals: 
1) find a robust solution too calculating omega, with appriopriate validation and falsification metrics. 
2) find an optimal pattern for gpu acceleration without relying on jax due too windows conflicts. 
3) ensure mathematical fidelity and appriopriate test measures too avoid false positives. 

expected lead time- 1 week. 
