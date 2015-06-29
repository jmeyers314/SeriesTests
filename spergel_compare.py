""" Compare MCMC samples obtained using galsim.Spergel profile to those obtained using
galsim.SpergelSeries profile.  We'll generate an image, (optionally adding noise, although I think
this might not be necessary), and then sample from the likelihood P(image | params).  The outputs
include triangle plots comparing the samples and moments up to all possible 4th central moments of
the 6 parameters defining the galaxy images.
"""

import cPickle
import galsim
import emcee
import numpy as np
import triangle
import matplotlib.pyplot as plt
import itertools

def moments(flatchain):
    nsample, nparam = flatchain.shape
    means = np.mean(flatchain, axis=0)
    centered = flatchain - means
    out = {}
    for i in range(nparam):
        out[(i,)] = means[i]
    for npt in range(2, 5):
        for c in itertools.combinations_with_replacement(range(nparam), npt):
            data = np.ones(nsample, dtype=float)
            for i in c:
                data *= centered[:,i]
            out[c] = np.mean(data)
    return out

def getimage(p, args, do_series):
    x0, y0, HLR, flux, e1, e2 = p
    psf = galsim.Moffat(beta=3, fwhm=args.PSF_FWHM)
    if do_series:
        gal = galsim.SpergelSeries(nu=args.nu, jmax=args.jmax, half_light_radius=HLR, flux=flux)
    else:
        gal = galsim.Spergel(nu=args.nu, half_light_radius=HLR, flux=flux)
    gal = gal.shear(e1=e1, e2=e2)
    gal = gal.shift(x0, y0)
    if do_series:
        final = galsim.SeriesConvolution(gal, psf)
    else:
        final = galsim.Convolve(gal, psf)
    img = final.drawImage(nx=args.nx, ny=args.ny, scale=args.scale)
    return img

def lnprob(p, target_image, noisevar, args, do_series):
    #print p
    x0, y0, HLR, flux, e1, e2 = p
    if (HLR < 0.01
        or HLR > 10
        or flux < 0.01
        or flux > 300
        or e1**2 + e2**2 > 1.0
        or abs(x0) > 3
        or abs(y0) > 3):
        return -np.inf
    try:
        img = getimage(p, args, do_series)
    except RuntimeError:
        return -np.inf
    return np.sum(-(img.array - target_image.array)**2 / (2.0 * noisevar))

def sample(args, do_series=False):
    """
    Generate MCMC (emcee) samples from arguments specified in args.

    @param series  Boolean controlling whether or not to use series approx.

    @returns post-sampling emcee sampler object.
    """
    # 3 random number generators to seed
    bd = galsim.BaseDeviate(args.image_seed) # for GalSim
    rstate = np.random.mtrand.RandomState(args.sample_seed + args.jmax).get_state() # for emcee
    np.random.seed(args.sample_seed + args.jmax) # for numpy functions called outside of emcee

    nwalkers = args.nwalkers
    nsteps = args.nsteps
    ndim = 6

    p_initial = [args.x0, args.y0, args.HLR, args.flux, args.e1, args.e2]
    #            x0, y0,  HLR, flux, e1, e2
    p_std = [0.01, 0.01, 0.01, args.flux*0.01, 0.01, 0.01]
    x0, y0, HLR, flux, e1, e2 = p_initial

    psf = galsim.Moffat(beta=3, fwhm=args.PSF_FWHM)
    gal = galsim.Spergel(nu=args.nu, half_light_radius=args.HLR, flux=args.flux)
    gal = gal.shear(e1=args.e1, e2=args.e2)
    gal = gal.shift(args.x0, args.y0)
    final = galsim.Convolve(gal, psf)
    target_image = final.drawImage(nx=args.nx, ny=args.ny, scale=args.scale)

    noise = galsim.GaussianNoise(rng=bd)
    if args.noisy_image:
        noisevar = target_image.addNoiseSNR(noise, args.SNR, preserve_flux=True)
    else:
        dummy_image = target_image.copy()
        noisevar = dummy_image.addNoiseSNR(noise, args.SNR, preserve_flux=True)
    p0 = np.empty((nwalkers, ndim), dtype=float)
    todo = np.ones(nwalkers, dtype=bool)
    lnp_args = [target_image, noisevar, args, do_series]
    while len(todo) > 0:
        p0[todo] = [p_initial + p_std*np.random.normal(size=ndim) for i in range(len(todo))]
        todo = np.nonzero([not np.isfinite(lnprob(p, *lnp_args)) for p in p0])[0]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnp_args)
    sampler.run_mcmc(p0, nsteps, rstate0=rstate)
    return sampler

def report(chain):
    N = chain.shape[0] * chain.shape[1]
    print "Results"
    print "-------"
    print "x0 = {:6.4f} +/- {:6.4f}".format(np.mean(chain[...,0]), np.std(chain[..., 0]))
    print "y0 = {:6.4f} +/- {:6.4f}".format(np.mean(chain[...,1]), np.std(chain[..., 1]))
    print "HLR = {:6.4f} +/- {:6.4f}".format(np.mean(chain[...,2]), np.std(chain[..., 2]))
    print "flux = {:6.2f} +/- {:6.2f}".format(np.mean(chain[...,3]), np.std(chain[..., 3]))
    print "e1 = {:6.4f} +/- {:6.4f}".format(np.mean(chain[...,4]), np.std(chain[..., 4]))
    print "e2 = {:6.4f} +/- {:6.4f}".format(np.mean(chain[...,5]), np.std(chain[..., 5]))

# def shade(ax, nburn):
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     ax.fill_between([xlim[0], nburn], [ylim[0]]*2, [ylim[1]]*2, color='k', alpha=0.3)
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)

def plot(spergel_sampler, series_samplers, args):
    p_initial = [args.x0, args.y0, args.HLR, args.flux, args.e1, args.e2]
    ndim = spergel_sampler.chain.shape[-1]

    # triangle plots
    spergel_samples = spergel_sampler.chain[:, args.nburn:, :].reshape((-1, ndim))

    extents = []
    for i in range(ndim):
        vals = spergel_samples[:,i]
        for jmax in range(args.jmaxmin, args.jmaxmax+1):
            series_samples = series_samplers[jmax].chain[:, args.nburn:, :].reshape((-1, ndim))
            vals = np.concatenate([vals, series_samples[:, i]])
        extents.append(np.percentile(vals, [0.5, 99.5]))

    for jmax in range(args.jmaxmin, args.jmaxmax+1):
        series_samples = series_samplers[jmax].chain[:, args.nburn:, :].reshape((-1, ndim))
        labels = ["x0", "y0", "HLR", "flux", "e1", "e2"]
        fig = triangle.corner(spergel_samples, labels=labels, truths=p_initial, extents=extents)
        fig = triangle.corner(series_samples, color='red', extents=extents, fig=fig)

        fig.savefig(args.plot_prefix+"_jmax_{:02d}_triangle.png".format(jmax))
        plt.close(fig)

    # # parameter traces
    # fig = plt.figure()
    # for i in range(6):
    #     ax = fig.add_subplot(3, 2, i+1)
    #     ax.plot(sampler.chain[..., i].T, alpha=0.3)
    #     ax.set_ylabel(labels[i])
    #     shade(ax, args.nburn)
    #     ax.axhline(p_initial[i], color='k')
    # fig.set_tight_layout(True)

    # # lnprob trace
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(sampler.lnprobability.T, alpha=0.3)
    # ax.set_ylabel("lnprob")
    # shade(ax, args.nburn)

    # # sample images
    # bd = galsim.BaseDeviate(args.seed) # for GalSim
    # psf = galsim.Moffat(beta=3, fwhm=args.PSF_FWHM)
    # if args.spergel or args.spergelseries:
    #     gal = galsim.Spergel(nu=args.nu, half_light_radius=args.HLR, flux=args.flux)
    # else:
    #     gal = galsim.Gaussian(half_light_radius=args.HLR, flux=args.flux)
    # gal = gal.shear(e1=args.e1, e2=args.e2)
    # gal = gal.shift(args.x0, args.y0)
    # final = galsim.Convolve(gal, psf)
    # target_image = final.drawImage(nx=args.nx, ny=args.ny, scale=args.scale)

    # noise = galsim.GaussianNoise(rng=bd)
    # noisevar = target_image.addNoiseSNR(noise, args.SNR, preserve_flux=True)

    # fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1)
    # vmin = target_image.array.min()
    # vmax = target_image.array.max()
    # ax.imshow(target_image.array, vmin=vmin, vmax=vmax)

    # ax = fig.add_subplot(2, 1, 2)
    # i = np.argmax(sampler.flatlnprobability)
    # p = sampler.flatchain[i]
    # img = getimage(p, args)
    # ax.imshow(img.array, vmin=vmin, vmax=vmax)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nwalkers", type=int, default=50,
                        help="Default: 50")
    parser.add_argument("--nsteps", type=int, default=200,
                        help="Default: 200")
    parser.add_argument("--nburn", type=int, default=100,
                        help="Default: 100")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--plot_prefix", type=str, default="spergel_compare",
                        help="Default: spergel_compare")
    parser.add_argument("--outfn", type=str,
                        help="Default: None")

    parser.add_argument("--x0", type=float, default=0.0,
                        help="Default: 0.0")
    parser.add_argument("--y0", type=float, default=0.0,
                        help="Default: 0.0")
    parser.add_argument("--HLR", type=float, default=0.2,
                        help="Default: 0.2")
    parser.add_argument("--flux", type=float, default=100.0,
                        help="Default: 100")
    parser.add_argument("--e1", type=float, default=0.0,
                        help="Default: 0.0")
    parser.add_argument("--e2", type=float, default=0.0,
                        help="Default: 0.0")

    parser.add_argument("--SNR", type=float, default=40.0,
                        help="Default: 40")
    parser.add_argument("--noisy_image", action='store_true',
                        help="Default: unset")
    parser.add_argument("--nx", type=int, default=32,
                        help="Default: 32")
    parser.add_argument("--ny", type=int, default=32,
                        help="Default: 32")
    parser.add_argument("--scale", type=float, default=0.2,
                        help="Default: 0.2")
    parser.add_argument("--sample_seed", type=int, default=1234,
                        help="Default: 1234")
    parser.add_argument("--image_seed", type=int, default=1234,
                        help="Default: 1234")

    parser.add_argument("--PSF_beta", type=float, default=0.3,
                        help="Default: 3.0")
    parser.add_argument("--PSF_FWHM", type=float, default=0.7,
                        help="Default: 0.7")

    parser.add_argument("--nu", type=float, default=0.0,
                        help="Default: 0.0")

    parser.add_argument("--jmaxmin", type=int,
                        help="Default: None")
    parser.add_argument("--jmaxmax", type=int,
                        help="Default: None")
    parser.add_argument("--jmax", type=int, default=7,
                        help="Default: 7")

    args = parser.parse_args()

    # Sample first using exact Spergel profile
    spergel_sampler = sample(args, do_series=False)
    spergel_moments = moments(spergel_sampler.flatchain)
    print
    print "Exact profile:"
    report(spergel_sampler.chain)

    # Now sample SpergelSeries profiles
    series_samplers = {}
    series_momentss = {}
    if args.jmaxmin is None:
        args.jmaxmin = args.jmaxmax = args.jmax
    for jmax in range(args.jmaxmin, args.jmaxmax+1):
        args.jmax = jmax
        series_samplers[jmax] = sample(args, do_series=True)
        series_momentss[jmax] = moments(series_samplers[jmax].flatchain)
        print
        print "Series profile (jmax={})".format(jmax)
        report(series_samplers[jmax].chain)

    if args.plot:
        plot(spergel_sampler, series_samplers, args)

    if args.outfn is not None:
        out = {'args':args,
               'spergel_moments':spergel_moments,
               'series_momentss':series_momentss}
        cPickle.dump(out, open(args.outfn, 'w'))