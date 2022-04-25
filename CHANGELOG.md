# Changelog

## [0.6.3] - 2022-04-26
### Added
- Added friendly __str__ for samplers.
- Added KFold CV option to KDE bandwidth selection for ResampleKde.
- Added BIC scan option for number of component selection in ResampleGmm.
- Example notebooks in the documentation
## Fixed
- Refactored "Resample" to "Sample", align with
- SamplerHist interface, moved "replace "arg to constructor.
- SamplerHist allow for repeated samples in case the number of
  drawn samples is larger than the source set.

## [0.6.2] - 2022-04-26

### Fixed
- Documentation and coverage


## [0.6.0] - 2022-04-25

### Fixed
- version releases again


## [0.5.0] - 2022-04-26

### Added
- Generic Resample class.
### Fixed
- version releases


## [0.4.0] - 2022-04-25

### Added
- two-sample Kolmogorov–Smirnov test.
- two-sample Anderson–Darling test.
