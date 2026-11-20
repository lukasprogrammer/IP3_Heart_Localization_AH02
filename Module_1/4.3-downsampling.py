def downsampling(Fs, av, mv, pv, tv):


    M = Fs/4000

    x_new = x[::M]