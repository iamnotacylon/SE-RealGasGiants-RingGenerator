import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import erf
import random


min_r=0
loop_counter=0

#########user-tunable ring generation constants#########

image_height = 2048
image_width = 4096

######global ring parameters######
max_inner_edge=100
max_outer_edge=image_height

######ring color parameters######
color_lb=230
color_ub=240
banding_noise_mean=0
banding_noise_stdev=3 #out of 255 - this is the stdev of the gaussian noise being applied to the colors (in the red band, specifically), so this value squared will be the average difference from the "baseline" ring color
random_noise_stdev=1 # out of 255 - this is the stdev of the gaussian noise being applied to all pixels in the image, not just horizontal bands

######ring width and spacing parameters######
ring_min_gap_width=10
ring_max_gap_width=150 #set to -1 for sparser generation style
ring_min_width=10
ring_max_width=500 #the rng is supposed to return a uniform distribution, so the mean should be (min+max)/2 ; the min here is 10 and the max shrinks down below as constrained by an envelope function
ring_max_width_for_fuzzy_edges=50

#ring_max_width envelope function parameters
ring_max_width_beta=1/120
ring_max_width_offset=1450

######Sub-Ring Parameters###########
sub_ring_exponent=-3 # more negative makes the sub-rings fainter.

######predefined ring gaps######
# list of [start, end] for gaps
# defined_gaps=[[689-25,689+25], [1178-150, 1178+150], [1587-50, 1587+50], [1689-50,1689+50], [1894-50, max_outer_edge]]
defined_gaps = []

########################################################

def make_rings(dx, dr, rng):
    x = np.arange(0., image_height)
    params = _make_params(dx, dr, rng, x)
    y = multiples(x, *params)
    return y/np.max(y)

def _make_params(dx, dr, rng, x, params=None, depth=0):
    if params is None:
        params = []

    mu = dx+dr/2
    # decreasing additional amplutide for each sub-ring depth
    a = 1+depth
    a**=sub_ring_exponent
    if depth > 0:
        a*=rng.random()
    # Fuzzy smaller rings, but not the sub rings
    fuzzy = dr < ring_max_width_for_fuzzy_edges and depth==0
    sigma = max(dr/2.5,ring_min_width) if fuzzy else max(1,dr/25.)

    # Add our parameters to the stripes
    params.append(a)
    params.append(sigma)
    params.append(mu)
    params.append(dr/2) # shifts for erf if needed, +- this amount
    
    # Make sub rings
    ring_width_frac = dr/ring_max_width
    num_subrings=round(rng.uniform(0, 30*ring_width_frac))
    outer_edge = dx+dr
    sub_ring_min = ring_min_width/5
    sub_ring_max = max(int(dr/5), ring_min_width*5)
    for _ in range(num_subrings):
        _dx = rng.randint(dx, outer_edge-sub_ring_min)
        finished_subring_inner=False
        while not(finished_subring_inner):
            _dr = rng.randint(sub_ring_min, sub_ring_max)
            if _dx+_dr > outer_edge:
                continue
            finished_subring_inner=True
        params = _make_params(_dx, _dr, rng, x, params, depth+1)

    return params

# This function is a any number of rings
def multiples(x, *params):
    y = np.zeros(len(x))
    for i in range(0, len(params), 4):
        a = params[i]
        sigma = params[i+1]
        mu = params[i+2]
        dr = params[i+3]
        if sigma < dr/3 or sigma <= 2:
            y = y + flatter_ring(x, a, sigma, mu, dr)
        else:
            y = y + gaussian(x, a, sigma, mu)
    return y

def gaussian(x, a, sigma, mu):
    dx = x-mu
    dx2 = dx*dx
    s2 = sigma*sigma
    return a*np.exp(-dx2/(2*s2))

def flatter_ring(x, a, sigma, mu, dr):
    y_min = 1-erf((x-mu-dr)/(math.sqrt(2)*sigma))
    y_max = 1-erf(-(x-mu+dr)/(math.sqrt(2)*sigma))
    return a*y_min*y_max*0.25 # the above y_max and y_min vary from 0 to 2, so *0.25 here.

def compl_logistic_func(x, beta, offset):
    return 1 - (1 / (1+np.exp(-beta*(x-offset))))

def gen_ring_file(name, rng_v): #predef_gaps contains inner radii in left column and outer radii in right column
    img = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    rng = random.Random(rng_v)
    print(f"Starting {name}")

    min_r=0
    loop_counter=0
    finished_outer=False
    rings = []

    if len(defined_gaps)>0:
        defined_gaps.insert(0, [0, max_inner_edge])
        for i in range(len(defined_gaps)-1):
            dx = defined_gaps[i][1]
            dr = defined_gaps[i+1][0] - dx
            rings.append([dx, dr])
    else:
        while not(finished_outer):
            dx = rng.randint(min_r, max_outer_edge) #generate ring starting radius

            #if any of these are true about dx, it starts the while loop over and makes a new dx
            if loop_counter==0 and dx>max_inner_edge: #if this the first loop, are we within tolerance for the max inner edge radius?
                continue
            if ring_max_gap_width!=-1 and dx-min_r>ring_max_gap_width: #is ring starting radius within the specified distance of the previous ring?
                continue

            loop_counter+=1
            
            #Generate Ring Ending Radius, with an Envelope Function that makes rings narrower near the outer edge
            a=compl_logistic_func(dx, ring_max_width_beta, ring_max_width_offset)
            current_ring_max_width = max(ring_min_width+1, round(a*ring_max_width))
            finished_inner=False
            while not(finished_inner):
                dr = rng.randint(ring_min_width, current_ring_max_width) #generate ring width
                if dx + dr > max_outer_edge: #Ensure ring fits in the image
                    continue 
                finished_inner=True

            rings.append([dx, dr])

            min_r=dx+dr+ring_min_gap_width
            if min_r>max_outer_edge-200:
                finished_outer=True

    # Pack all at once
    ring_info = open(f"{name}.txt", 'w')
    print(f"Saving Ring {name}")
    x = np.arange(0., image_height)
    y = np.zeros_like(x)
    ring_zones = ""
    for ring in rings:
        dx = ring[0]
        dr = ring[1]
        _ring = make_rings(dx, dr, rng)
        y += _ring
        ring_info.write(f"{dx},{dx+dr}\n")
        if ring_zones == "":
            ring_zones=f"{dx/image_height:.3f},{(dx+dr)/image_height:.3f}"
        else:
            ring_zones=f"{ring_zones},{dx/image_height:.3f},{(dx+dr)/image_height:.3f}"

    ring_info.write(ring_zones)
    ring_info.close()
    y /= np.max(y)
    # plt.plot(x, y)
    # plt.show()
    img = gen_ring_bitmap(rings, img, rng, alphaArr=y)

    img = np.ascontiguousarray(img[::-1]) # img=np.flip(img, axis=0) #flips to orient the same way as the mod expects the texture to go (bottom of image is next to planet, top of image is outer edge)
    plt.imsave(f"{name}.png", img)
    print(f"Saved Ring {name}")

def gen_ring_bitmap(rings, img, rng, alphaArr):

    dx = 0
    dr = 2048
    colours = []
    ring_mids = []
    ring_noise = []

    # Init colours and band noise for the rings
    for ring in rings:
        _dx = ring[0]
        _dr = ring[1]
        _cr = rng.randint(color_lb, color_ub) # red value for ring
        _cg = rng.randint(color_lb, color_ub) # green value for ring
        _cb = rng.randint(color_lb, color_ub) # blue value for ring
        colours.append([_cr,_cg,_cb])
        ring_mids.append(_dx+_dr/2)
        noise=np.random.normal(banding_noise_mean, banding_noise_stdev, size=_dr)
        arrays_r = [noise for _ in range(image_width)]
        noise=np.random.normal(banding_noise_mean, banding_noise_stdev, size=_dr)
        arrays_g = [noise for _ in range(image_width)]
        noise=np.random.normal(banding_noise_mean, banding_noise_stdev, size=_dr)
        arrays_b = [noise for _ in range(image_width)]
        arrays_r = np.stack(arrays_r, axis=1)
        arrays_g = np.stack(arrays_g, axis=1)
        arrays_b = np.stack(arrays_b, axis=1)
        ring_noise.append([arrays_r,arrays_g,arrays_b])

    ring_mids = np.array(ring_mids)

    # Generate each line on the image, based on closest ring
    for i in range(dr):
        x = dx + i
        ring_i = np.argmin(np.abs(ring_mids-x))
        cr,cg,cb = colours[ring_i]
        # Init colour to a random noisy colour based on closest ring
        noise = np.array([
            np.random.normal(cr, random_noise_stdev, size=image_width),
            np.random.normal(cg, random_noise_stdev, size=image_width),
            np.random.normal(cb, random_noise_stdev, size=image_width),
            np.ones((image_width))*255,
        ]).T
        img[x] = noise

    # Then add each set of ring dependent banding
    for ring_i in range(len(rings)):
        _dx, _dr = rings[ring_i]
        arrays_r,arrays_g,arrays_b = ring_noise[ring_i]
        img[_dx:_dx+_dr,:,0] = img[_dx:_dx+_dr,:,0]+arrays_r
        img[_dx:_dx+_dr,:,1] = img[_dx:_dx+_dr,:,1]+arrays_g
        img[_dx:_dx+_dr,:,2] = img[_dx:_dx+_dr,:,2]+arrays_b

    alphaMask = np.broadcast_to(alphaArr, img.T.shape).T
    img = img*alphaMask
    img = img.astype(np.uint8)
    return img

if __name__ == "__main__":
    rng_v = random.random()
    rng_v = 0
    rng = random.Random(rng_v)

    # for x in range(5):
    #     rng_v = rng.random()
    #     gen_ring_file(f"ring_{x}",rng_v)

    import multiprocessing
    import time
    processes = []
    for x in range(5):
        rng_v = rng.random()
        p = multiprocessing.Process(target=gen_ring_file, args=(f"ring_{x}",rng_v,))
        p.start()
        processes.append(p)
        time.sleep(1)
        while len(processes)>5:
            p = processes[0]
            p.join()
            processes.remove(p)
    for p in processes:
        p.join()