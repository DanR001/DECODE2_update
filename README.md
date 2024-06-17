=================================================

# DECODE: Discrete statistical sEmi-empiriCal mODEl

=================================================


## Version 2 of DECODE.

## Authors:
Hao Fu <h.fu@soton.ac.uk>
Lumen Boco <lboco@sissa.it>
Daniel Roberts <d.m.roberts@soton.ac.uk>
Cressida Cleland <cleland@apc.in2p3.fr>

## The purpose of this fork:
  - Updating the BH recipe, the abundance matching to have sHAR-sSFR as an option, and a general tidy up.

## To Do List:
  - Finish testing the complete HAR function using the complete sample from SatGen
  - Fix the problem with sHAR - sSFR abundance matching
  - Finish the class to have standardised methods for analysis of the catalogues
  - Check if the difference between BH map and dNac is a bug or an intrinsic difference of the methods
  - Fix the reactivate option in ntegrate_BHAR
  - Finish testing exponential quenching in all
  - Implement Sub Halo evolution between infall and merging
  - Implement BH evolution between infall and merging
  - Add in the possibility of adding an additional BH merging time scale
      - Assume after t_dyn the BHs are in a binary
      - Use the binary hardening scale from Yu (2002) and assume they are at mpc scale after this 
      - Sample an initial semi-major axis and initial orbital eccentricity
      - Use the Peters (1964) prescription for the merging time due to GW emiision alone
  - Update / create a new version of generate SG merger tree to get N merger trees of uniform mass distribution using sat gen and then scale the contribution of these to the HAR/sHAR function by scaling contribution using the HMF
  - 
