"""Temperature-gradient template handling and dTemp measurements."""

import glob
import os

import numpy as np
from astropy.io import fits

from calcspec import barshift
from gplot import gplot
from pause import pause
from read_spec import Spectrum, is_serval_tpl, read_template


def _split_1d_template_by_order(wave, flux, reference):
   wmin = np.nanmin(reference.w, axis=1)
   wmax = np.nanmax(reference.w, axis=1)
   limits = np.searchsorted(wave, [wmin, wmax])
   flux = [flux[slice(*limit)] for limit in limits.T]
   wave = [wave[slice(*limit)] for limit in limits.T]

   for order in range(len(wave)):
      if len(wave[order]) == 0 or np.all(np.isnan(flux[order])):
         wave[order] = 0 * reference.w[order]
         flux[order] = 0 * reference.f[order]
   return wave, flux


def _template_paths(template_input):
   paths = template_input if isinstance(template_input, list) else [template_input]
   output = []
   for path in paths:
      if os.path.isdir(path):
         template_file = path + os.sep + os.path.basename(path.rstrip(os.sep)) + '.fits'
         if os.path.exists(template_file) and is_serval_tpl(template_file):
            output += [path]
         else:
            output += sorted(glob.glob(path + os.sep + '*.fits'))
      else:
         output += [path]
   return output


def _read_template(path, reference, inst, drs, fib, targ):
   if path.endswith('.s1d.fits'):
      with fits.open(path) as hdu:
         wave = hdu[1].data['lnwave'].astype(float)
         flux = hdu[1].data['flux'].astype(float)
      wave, flux = _split_1d_template_by_order(wave, flux, reference)
      return wave, flux, None

   if is_serval_tpl(path):
      filename = path + os.sep + os.path.basename(path.rstrip(os.sep)) + '.fits' if os.path.isdir(path) else path
      wave, flux, quality, _ = read_template(filename)
      return wave, flux, quality, None

   try:
      spectrum = Spectrum(path, inst=inst, pfits=True, orders=np.s_[:], drs=drs, fib=fib, targ=targ)
      quality = 1. * (spectrum.bpmap == 0)
      return barshift(spectrum.w, spectrum.berv), spectrum.f, quality
   except Exception as error:
      raise ValueError('Unsupported dTemp template format: %s (%s)' % (path, error))


def _mean_templates(template_sets, nord):
   reference_wave, reference_gradient, reference_bad = template_sets[0]
   if len(template_sets) == 1:
      return reference_wave, reference_gradient, reference_bad

   gradient = []
   bad = []
   for order in range(nord):
      values = []
      for wave, order_gradient, _, _ in template_sets:
         if len(wave[order]) == 0:
            continue
         values += [np.interp(reference_wave[order], wave[order], order_gradient[order], left=np.nan, right=np.nan)]
      gradient += [np.nanmean(values, axis=0) if values else np.nan * reference_wave[order]]
      if any(order_bad is not None for _, _, order_bad, _ in template_sets):
         bad_values = [np.interp(reference_wave[order], wave[order], 1. * (order_bad[order] > 0), left=1, right=1) > 0.01
                       for wave, _, order_bad, _ in template_sets if order_bad is not None]
         bad += [np.any(bad_values, axis=0)]
      else:
         bad += [None]
   return reference_wave, gradient, bad


def compute_gradient(template_input, rv_templates, reference, tpl_class, spline_cv, spline_ev,
                  tplvsini, velocity_range, tplqmin,
                  temperature_step, inst, drs, fib, targ):
   """Load one or more gradient/comparison spectra as per-order dTemp templates."""
   paths = _template_paths(template_input)
   if not paths:
      raise ValueError('No dTemp template fits files found in: %s' % template_input)

   template_sets = []
   for path in paths:
      print('restoring dTemp template:', path)
      wave, flux, quality = _read_template(path, reference, inst, drs, fib, targ)
      bad = [None] * len(flux) if quality is None else quality < tplqmin
      gradient = []
      for order, (order_wave, order_flux) in enumerate(zip(wave, flux)):
         if rv_templates[order] is None:
            gradient += [np.nan * order_flux]
         else:
            gradient += [(order_flux - rv_templates[order](order_wave)) / temperature_step]
      template_sets += [(wave, gradient, bad)]

   wave, gradient, bad = _mean_templates(template_sets, len(rv_templates))

   v_lo, v_hi = velocity_range
   return [None if rv_templates[order] is None else
           tpl_class(order_wave, order_gradient, spline_cv, spline_ev, bk=order_bad,
                     vsini=tplvsini, mask=True, vrange=[v_lo, v_hi])
           for order, (order_wave, order_gradient, order_bad) in enumerate(zip(wave, gradient, bad))]


def measure(residual, error, gradient, keep):
   """Project spectral residuals onto dA/dT and return dTemp and its error."""
   keep = np.asarray(keep)
   valid = (np.isfinite(residual[keep]) & np.isfinite(error[keep]) &
            np.isfinite(gradient[keep]) & (error[keep] > 0))
   keep = keep[valid]
   if not keep.size:
      return np.nan, np.nan

   weights = 1 / error[keep]**2
   denominator = np.dot(gradient[keep]**2, weights)
   if not np.isfinite(denominator) or denominator <= 0:
      return np.nan, np.nan

   temperature = np.dot(residual[keep] * gradient[keep], weights) / denominator
   normalized_residual = (residual[keep] - temperature * gradient[keep]) / error[keep]
   rms = np.sqrt(np.mean(normalized_residual**2))
   return temperature, np.sqrt(1 / denominator) * rms


def plot_measurement(wave, residual, error, gradient, keep, temperature,
                     temperature_error, obj, spectrum_number, order):
   """Plot the residual projection used for one per-order dTemp value."""
   model = temperature * gradient
   gplot.key('left Left rev samplen 2 title "%s (n=%s, o=%s, dTemp=%.3f+/-%.3f K)" noenhanced' %
             (obj, spectrum_number, order, temperature, temperature_error))
   gplot.xlabel('"wavelength"').ylabel('"flux residual"')
   gplot(wave, residual, error,
         'us 1:2:3 w e pt 7 ps 0.4 lc "grey" t "residuals (all)",',
         wave[keep], residual[keep],
         'us 1:2 w p pt 7 ps 0.5 lc "red" t "residuals (used)",',
         wave, model,
         'us 1:2 w l lw 2 lc "black" t "dTemp model"')
   pause('dTemp', order, '%.3f +/- %.3f K' % (temperature, temperature_error))
   gplot.reset()