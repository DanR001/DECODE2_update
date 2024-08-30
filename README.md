=================================================

# DECODE: Discrete statistical sEmi-empiriCal mODEl

=================================================


## Version 2 of DECODE.

## Authors:
Hao Fu <h.fu@soton.ac.uk>
Lumen Boco <lboco@sissa.it>
Daniel Roberts <d.m.roberts@soton.ac.uk>
Cressida Cleland <cleland@apc.in2p3.fr>
Francesco Shankar <f.shankar@soton.ac.uk>

## The purpose of this fork:
  - Rewrite to now loop over z, rather than calculating each quantity separately at all z 
  - Updating the BH recipe, the abundance matching to have sHAR-sSFR as an option, and a general tidy up.

## To Do List:
  - Fix the problem with sHAR - sSFR abundance matching
  - Implement Sub Halo evolution between infall and merging
  - Implement BH evolution between infall and merging
  - Add in the possibility of adding an additional BH merging time scale
      - Assume after t_dyn the BHs are in a binary
      - Use the binary hardening scale from Yu (2002) and assume they are at mpc scale after this 
      - Sample an initial semi-major axis and initial orbital eccentricity
      - Use the Peters (1964) prescription for the merging time due to GW emiision alone
