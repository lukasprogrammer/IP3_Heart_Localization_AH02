def downsampling(Fs, av, mv, pv, tv):

    M = Fs/4000

    av_new = av[::M]
    mv_new = mv[::M]
    pv_new = pv[::M]
    tv_new = tv[::M]

    return [av_new, mv_new, pv_new, tv_new]