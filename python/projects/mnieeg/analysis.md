## Sensor Level

We observed

Channels 50 and 52 were corrected by replacing the x coordinate with the mean of [49, 51] and [51, 53], respectively. Channel 22 was corrected by replacing the x and y coordinate with the means of [21, 23]. Finally, all channels were (re)projected onto the MNI skin surface.


#### Error in Mean and Standard Deviation

It seems that electrode errors are more pronounced in occipital and lateral regions.


#### ABSOLUTE ERROR ALONG EACH AXIS

The errors seem mostly in the y (top) and z (posterior and lateral) directions and not so much in x suggesting that the placement along AP is more difficult compared LR.


## Forward Level

In general, it seems that RDM is sensitive to errors in sensor position whereas lnMAG is more sensitive to errors in the geometry of the head model.

The magnitudes from a template model cannot be trusted, it seems.

- How do these relate to those in the Vorwerk 2014 paper? In terms of spatial distribution and size of the effects.


#### Heat maps of distances (mean RDM/abs(lnMAG)) between forward models for models/subjects

Generally, for a method to "make sense" we should see an effect of subject (i.e., lower values on the diagonal). Otherwise, why model the individual anatomy/sensor layout at all.

###### RDM
Generally effects of subject and forward model are discernible. The diagonal blocks (intra-model RDMs) show that there are generally more variation in the cases where the digitized positions are used. The variation within custom_nonlin is small suggesting a fairly consistent electrode placement (which may not correspond to reality though?!) more so than the manufacturer_affine_lm probably because they may be quite far from the skin before being projected onto it (in this model they may get projected quite differently).
One subject (sub-12) seems different from the rest in terms of elec. pos. and/or anatomy. (the effect is not seen in manufacturer_affine_lm vs. digitized suggesting that the former is so crude in all cases that this may not be worse than other cases) and that it is probably more due to positions than anatomy. (sub-12 the cap is shifted quite a lot backwards!)

###### abs(lnMAG)
no apparent effect of electrode placement but there is for the template: all template models are quite similar because the anatomy is almost identical and so there is no effect of subject (!) when comparing with digitized suggesting that the electrode positions do not play a major role in determining this but rather the anatomy/geometry of the head model (and conductivity of course but this is not varied here).


#### RDM

Generally seems to correlate very well the mean error in electrode positions such that areas where the error is large in the electrodes generally also have higher RDM (mostly occipital but also to some degree frontal)

[for the custom_nonlin layout there is higher RDM around electrode 22 and the two occipital ones where there seems to be a slight difference between the digitized template and the actual digitized positions. Not exactly sure why. EEG 22 may be an error but the occ. elecs are symmetric...]

errors larger on gyral crowns than sulcal walls and valleys so where the sensitivity is largest and if the sources are quasi-radial or not.

the same seems true for manufacturer_affine_lm but the errors are much larger in the occipital and some of the temporal area

Deep sources seem less affected (unless effects are large as in manufacturer_affine_lm!).

For template_nonlin, the mean RDM is smaller than that of the manufacturer_affine_lm! And close to the custom_nonlin. However, the tail is heavier suggesting some kind of outliers perhaps (where the fit between "average anatomy" and individual anatomy is bad?).

RDM seems larger in frontal-most PFC and temporal area and around V1. Perhaps the inter-individual anatomical variance (skull, air cavity, spongy bone, ...) is simply larger here than in other places?

Deep sources do not seem much affected.

#### lnMAG

##### custom_nonlin and manufacturer_affine_lm

generally increased in manufacturer_affine_lm in the inferior parts of the brain perhaps because the electrodes extends further down thus giving slightly better coverage to these areas (compared to the actual positions). On the other hand (and as a consequence) there is decreased sensitivity in the superior parts for manufacturer_affine_lm and also to some extent for custom_nonlin. This fits the above explanation because if the sensitivity is increased in inferior parts due to better coverage, the coverage must be worse in other parts, in particular, the superior ones as the electrodes are more thinly spread across this area... The effect is more dramatic with manufacturer_affine_lm.

The opposite effect is seen around V1 which is placed higher than inion the more inferior displacement may not be beneficial to this area because the skull gets thicker around this area...

this correlates with the differences in z coordinate for occipital, temporal, and frontal areas.

##### template_nonlin

The magnitude differences are much larger in the template.

The template shows increased sensitivity on the gyral and sulcal crowns (larger effect on the gyral crowns) and decreased sensitivity on the sulcal walls. Thus, sensitivity is increased to quasi-radial sources (orthogonal to the sensor array) whereas it is decreased to quasi-tangential sources (perpendicular to the sensor array).

The template head model generally has very little spongy bone. This is particularly prominent in occipital and superior areas. Around PFC there is an air cavity. Changes in skull shape most prominent around V1 and PFC?

In general the registration/warp of the template should be a little more accurate at the top of the head than the sides/front/back as there are not many points in these areas.


This (lack of spongy bone) may explain the decreased sensitivity to V1 and PFC as there is generally a fair amount of this in these regions in the actual head models. (at least in occ. area)

Slightly decreased in occipital and prefrontal areas. Skull thickness?

Differences are larger in left temporal cortex compared to right temporal cortex. Why?



## Inverse Level


Plot a PSF and a CTF? Or just refer to Hauk paper(s)?

Use peak_err or cog_err?? PLE has been used mostly before it seems and most people may just choose the max activation when localizing, however, CGE is more stable.

For overall effects, show histograms/cumulative histograms of SNR/INV/FWD: SNR along rows, INV along cols, FWD in the histograms.

PSF, PLE
PSF, SDE

(only MNE)
CTF, PLE
CTF, SDE


Does it make sense to compute RDM/lnMAG of the PSFs for the different forward models? E.g., use the PSFs of the digitized solution as "reference" and then compare with PSFs from the other electrode layouts? Digitized may not always be best in terms of PLE (but almost always in cog_err) [see below] so does it make sense to use as a reference? The "actual" reference would be a delta function on the relevant source, however, the RDMs/lnMAGs to this are so large compared to the differences between models.


#### SNR


PSF, peak_err : for MNE the tail gets less heavy with snr. same goes for dSPM (less dramatically though). In both mne/dspm the mode stays more or less the
same though. for sLORETA there is almost no effect of SNR it seems (regularization is included in the normalization... does this explain?)

PSF, sd_ext : snr=3 no effect of forward model but this becomes more apparent as snr increases (particularly for MNE). spread generally decreases with increasing snr (i.e., less regularization) which makes sense (again particularly for MNE when the forward model is relatively accurate---manufacturer_affine_lm is not much affected).



#### Inverse Operator

PSF, peak_err: sLORETA has the lowest error, the dSPM, the MNE for all snr levels.

psf
peak_err = [-2.6, 2.6]
sd_ext = [-1.4, 1.4]

ctf
peak_err = [-1.6, 1.6]
sd_ext = [-1.5, 1.5]


Overall:
- Effects of forward and snr are generally quite similar for dSPM and sLORETA.
- Effects of SNR are generally similar for PSFs and CTFs.

The {model} - digitized difference in ...

- `peak_err` for `custom` and `template` show only little dependence on SNR across all inverse operators whereas `manufacturer` do show effects of SNR for `MNE` and `dSPM`.
- `sd_ext` for  `custom` and `template` show little effect of SNR with `dSPM` and `sLORETA` but do show effects for `MNE`. `manufacturer` shows effects of SNR for all inverse operators. In general, effects are largest for `MNE`.

- `peak_err` for `custom` and `template` is generally low across all inverse operators (`sLORETA` shows slightly larger differences than `dSPM` but that is before `digitized` has zero `peak_err` by design). The PSFs of MNE for `manufacturer` show increased error (~2.5 cm) in deep areas (an effect which decreases with SNR) and occipital and parietal areas (an effect which increases with SNR). For dSPM, there are also increased error (~2.5 cm) in occipital and parietal areas, however, this effect decreases with SNR. The CTFs for `manufacturer` show increased errors (~1.5 cm) in occipital and parietal areas for all SNR levels whereas an increase in errors (up to ~1.5 cm) in frontal and deep areas are seen as SNR increases.
- `sd_ext` of `custom` and `template`



we use > and < to denote decrease and increase with SNR, respectively. - = no effect of SNR




<font size=2>

|         | PLE |                     | SDE |      |
|---|---|---|---|---|
|         | PSF         | CTF         | PSF         | CTF         |
|**custom** |         |             |             |             |             |
| MNE    | ... | Low (slight increase in deep areas with SNR) | general increase particularly in sulcal valleys with SNR around superior parts. Same in deep areas. | increase in frontal (gyri)/occ./temporal with increased SNR  |
| dSPM    | (slightly larger in occ. for low SNR) | N/A |             | N/A |
|sLORETA  | ... | N/A  |             | N/A |
|**manufacturer**|    |             |             |             |             |
| MNE    | High in deep areas (> SNR), High in occipital/parietal areas (< SNR) | high in occ./parietal across SNR, increase with SNR in frontal and deep areas. |             |             |
| dSPM    | high in occ/parietal areas for low SNR - decrease with SNR | N/A |             | N/A |
|sLORETA  | ... | N/A |             | N/A |
|**template**|        |             |             |             |             |
|  MNE    |  ...           | very similar to custom |             |             |
| dSPM    | (slightly larger in occ. for low SNR [a little more than 'custom') | N/A |             | N/A |
| sLORETA  | ... | N/A |             | N/A |

</font>


###### MNE

`PSF, peak_err`
custom
- generally low, no effect of SNR

manufacturer
- high in deep areas for low SNR - decrease with SNR
- high in occ/parietal areas for high SNR - increase with SNR

template
- generally low, no effect of SNR

`PSF, sd_ext`
custom
- general increase particularly in sulcal valleys with SNR around superior parts.
Same in deep areas.

manufacturer
- low snr: high in occ. Increases with SNR in occ./parietal/frontal.
- generally higher in deep areas than custom and template.

template
- medium in occ. -> decreases with SNR
- increases slightly in deep areas
- generally lower than custom (except in occ. low snr)

`CTF, peak_err`
custom
- generally low. Not much effect of SNR (slight increase in deep areas with SNR)

manufacturer
- high in occ./parietal across SNR
- increase with SNR in frontal and deep areas.

template
- very similar to custom

`CTF, std_ext`
custom
- increase in frontal (gyri)/occ./temporal with increased SNR

manufacturer
- high in occ. for low SNR. Increase in occ/front/parietal/temporal with
increase SNR (specially gyri)

template
- increase in occ./temporal with increased SNR

###### dSPM

`PSF, peak_err`
custom
- generally low, not much effect of SNR (slightly larger in occ. for low SNR)

manufacturer
- high in occ/parietal areas for low SNR - decrease with SNR

template
- generally low, not much effect of SNR (slightly larger in occ. for low SNR [a
little more than 'custom')

`PSF, sd_ext`
custom
- generally low. Not much effect of SNR (slight increase)

manufacturer
- high in occ., spreads to parietal with increasing SNR
- generally increase with SNR in all areas.

template
- generally no effect of SNR
- high in occ. (all SNRs)

###### sLORETA

`PSF, peak_err`
custom / manufacturer / template
- No effect of SNR
- Generally higher errors than dSPM (but that is because of zero loc. error) otherwise they look similar to dSPM

`PSF, sd_ext`
very much like dSPM:
- custom: Not much effect of SNR (slight increase - mostly in sulci)
- template: medium in occ. (all SNRs)



#### Forward Model




---


#### Testing magnitude errors in source estimates?
the magnitudes of the source estimates using template_nonlin do not seem to be worse than the others... how come?

in minimum L2 norm we are penalizing source variance so this probably limits the effects seen on the magnitudes (as small magnitudes are preferred in general) and would tend to distribute it to a slightly worse data fit instead..?


#### PLE is sometimes worse when using the correct positions compared to template positions !?

peak localization error: the PSFs for the digitized inverse are, generally, closer to I (or is it just the diagonal which is closer to all ones?), however, due to the variability in them, the peak may not be closer to the actual source than using another inverse (it is generally better for higher snr, however, it is not perfect at all).

For example, for subject 1

  peak_err(digitized) <= peak_err(custom_nonlin) in 77.6 %
  cog_err(digitized) <= cog_err(custom_nonlin) in 98.9 %

std:

  peak_err(digitized) / peak_err(custom_nonlin) = 1.1 / 1.8 (heavy tails)
  cog_err(digitized) / cog_err(custom_nonlin) = 0.6 / 0.7

The tail in peak_err is much more pronounced in individual subject space than the impression one may get when viewing group results on fsaverage because of the averaging happening in the mapping to fsaverage and, obviously, when averaging across subjects.

-> thus, one can expect much more variability in the localization results when using peak_err compared to a more robust localization procedure.