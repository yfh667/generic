def intra_or_inter(N,i,next):
    i_orbit = i//N
    next_orbit = next//N
    if i_orbit == next_orbit:
        return True #  intra
    else:
        return False#  inter
