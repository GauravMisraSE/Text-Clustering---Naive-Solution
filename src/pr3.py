import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, vstack
from matplotlib import pyplot as plt

def csr_read(fname, ftype="csr", nidx=1):
    with open(fname) as f:
        lines = f.readlines()

    if ftype == "clu":
        p = lines[0].split()
        nrows = int(p[0])
        ncols = int(p[1])
        nnz = long(p[2])
        lines = lines[1:]
        assert (len(lines) == nrows)
    elif ftype == "csr":
        nrows = len(lines)
        ncols = 0
        nnz = 0
        for i in xrange(nrows):
            p = lines[i].split()
            if len(p) % 2 != 0:
                raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(p)))
            nnz += len(p) / 2
            for j in xrange(0, len(p), 2):
                cid = int(p[j]) - nidx
                if cid + 1 > ncols:
                    ncols = cid + 1
    else:
        raise ValueError("Invalid sparse matrix ftype '%s'." % ftype)
    val = np.zeros(nnz, dtype=np.float)
    ind = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows + 1, dtype=np.long)
    n = 0
    for i in xrange(nrows):
        p = lines[i].split()
        for j in xrange(0, len(p), 2):
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j + 1])
            n += 1
        ptr[i + 1] = n

    assert (n == nnz)

    return csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.float)

fname = 'train.dat'
mat = csr_read(fname)
# mat = mat.todense()
print mat.shape
mean = mat.mean(axis=0) # numpy matrix
print mean

def eucl(mean, target):
    diff = np.subtract(mean, target)
    diff_square = np.square(diff)
    sum = np.sum(diff_square)
    return math.sqrt(sum)

def get_meandeviation(matrix, len, mean):
    mean_deviation = 0
    for x in range(len):
        r = matrix.getrow(x).toarray()
        dcl = eucl(mean, r)
        mean_deviation = mean_deviation + dcl
    return mean_deviation / len

def getpivots(meandev, matrix, mean, len):
    cl = 0;
    rl = 0
    for x in range(200, 300):
        r = matrix.getrow(x).todense()
        dcl = eucl(mean, r)
        if dcl <= meandev + 0.5 and dcl >= meandev - 0.5:
            cl = x
            break
    for x in range(len - 300, len - 200):
        r = matrix.getrow(x).todense()
        dcl = eucl(mean, r)
        if dcl <= meandev + 0.5 and dcl >= meandev - 0.5:
            rl = x
            break
    return (cl, rl)

def getclouds(matrix, cl, rl, len):
    cloudcl = []
    cloudrl = []
    right = matrix.getrow(rl).toarray()
    left = matrix.getrow(cl).toarray()
    for x in range(len):
        row = matrix.getrow(x).toarray()
        dcl = eucl(left, row)
        drl = eucl(right, row)
        if dcl > drl:
            cloudrl.append(x)
        else:
            cloudcl.append(x)
    return (cloudcl, cloudrl)

def gen_csr(rowlist):
    ini = mat.getrow(rowlist[0])
    for x in range(1, len(rowlist)):
        ext = mat.getrow(rowlist[x])
        ini = vstack([ini, ext])
    return ini

mean_dev_mat = get_meandeviation(mat, 8580, mean)

print mean_dev_mat
pivot_tuple = getpivots(mean_dev_mat, mat, mean, 8580)
cl = pivot_tuple[0]; rl = pivot_tuple[1]
print cl, rl
clouds_tuple = getclouds(mat, cl, rl, 8580)
cloudleft = clouds_tuple[0]
cloudright = clouds_tuple[1]

print len(cloudleft), len(cloudright)

cloudleft_csr = gen_csr(cloudleft); cloudright_csr = gen_csr(cloudright)

mean_cloudleft = cloudleft_csr.mean(axis=0)
mean_dev_cloudleft_csr = get_meandeviation(cloudleft_csr, 2782, mean_cloudleft)

print mean_dev_cloudleft_csr

mean_cloudright = cloudright_csr.mean(axis=0)
mean_dev_cloudright_csr = get_meandeviation(cloudright_csr, 5798, mean_cloudright)

print mean_dev_cloudright_csr

pivot_tuple_left = getpivots(mean_dev_cloudleft_csr, cloudleft_csr, mean_cloudleft, 2782)
cl = pivot_tuple_left[0]; rl = pivot_tuple_left[1]
print cl, rl

clouds_tuple_left = getclouds(cloudleft_csr, cl, rl, 2782)

cloudleft_left = clouds_tuple_left[0]
cloudleft_right = clouds_tuple_left[1]
print len(cloudleft_left), len(cloudleft_right)

cloudleft_left_csr = gen_csr(cloudleft_left);
cloudleft_right_csr = gen_csr(cloudleft_right)

mean_cloudleft_left = cloudleft_left_csr.mean(axis=0)
mean_dev_cloudleft_left_csr = get_meandeviation(cloudleft_left_csr, 835, mean_cloudleft_left)

print mean_dev_cloudleft_left_csr

mean_cloudleft_right = cloudleft_right_csr.mean(axis=0)
mean_dev_cloudleft_right_csr = get_meandeviation(cloudleft_right_csr, 1947, mean_cloudleft_right)

print mean_dev_cloudleft_right_csr

# *************************** LEFT DONE **********************************

mean_cloudright = cloudright_csr.mean(axis=0)
mean_dev_cloudright = get_meandeviation(cloudright_csr, 5798 , mean_cloudright)
pivot = getpivots(mean_dev_cloudright, cloudright_csr , mean_cloudright , 5798)
cl = pivot[0]; rl = pivot[1]
# print cl, rl

clouds = getclouds(cloudright_csr, cl, rl, 5798)
cloudright_left = clouds[0]
cloudright_right = clouds[1]
print len(cloudright_left), len(cloudright_right)

cloudright_left_csr = gen_csr(cloudright_left); cloudright_right_csr = gen_csr(cloudright_right)

mean_cloudright_left = cloudright_left_csr.mean(axis=0)
mean_dev_cloudright_left = get_meandeviation(cloudright_left_csr, 835, mean_cloudright_left)

print mean_dev_cloudright_left

mean_cloudright_right = cloudright_right_csr.mean(axis=0)
mean_dev_cloudright_right = get_meandeviation(cloudright_right_csr, 1947, mean_cloudright_right)

print mean_dev_cloudright_right
# ************************************* CONTINUE WITH CLOUDRIGHT_LEFT

pivot_tuple_cloudright_left = getpivots(mean_dev_cloudright_left, cloudright_left_csr, mean_cloudright_left, 1675)
cl = pivot_tuple_cloudright_left[0]; rl = pivot_tuple_cloudright_left[1]
print cl, rl

clouds_tuple_cloudright_left = getclouds(cloudright_left_csr, cl, rl, 1675)
cloudRL_L = clouds_tuple_cloudright_left[0]
cloudRL_R = clouds_tuple_cloudright_left[1]

print len(cloudRL_L), len(cloudRL_R)

cloudRL_L_csr = gen_csr(cloudRL_L); cloudRL_R_csr = gen_csr(cloudRL_R)

mean_cloudRL_L = cloudRL_L_csr.mean(axis=0)
mean_dev_cloudRL_L = get_meandeviation(cloudRL_L_csr, 1006, mean_cloudRL_L)

print mean_dev_cloudRL_L

mean_cloudRL_R = cloudRL_R_csr.mean(axis=0)
mean_dev_cloudRL_R = get_meandeviation(cloudRL_R_csr, 669, mean_cloudRL_R)

print mean_dev_cloudRL_R
# continue with cloudright_right
pivot = getpivots(mean_dev_cloudright_right, cloudright_right_csr, mean_cloudright_right, 4123)
cl = pivot[0]; rl = pivot[1]
print cl, rl

clouds = getclouds(cloudright_right_csr, cl, rl, 4123)
cloudRR_L = clouds[0]
cloudRR_R = clouds[1]

print len(cloudRR_L), len(cloudRR_R)

cloudRR_L_csr = gen_csr(cloudRR_L); cloudRR_R_csr = gen_csr(cloudRR_R)

mean_cloudRR_L = cloudRR_L_csr.mean(axis=0)
mean_dev_cloudRR_L = get_meandeviation(cloudRR_L_csr, 2328, mean_cloudRR_L)

print mean_dev_cloudRR_L

mean_cloudRR_R = cloudRR_R_csr.mean(axis=0)
mean_dev_cloudRR_R = get_meandeviation(cloudRR_R_csr, 1795, mean_cloudRR_R)

print mean_dev_cloudRR_R
# continue with cloudRR_R
pivot = getpivots(mean_dev_cloudRR_R, cloudRR_R_csr, mean_cloudRR_R, 1795)
cl = pivot[0]; rl = pivot[1]
print cl, rl

clouds = getclouds(cloudRR_R_csr, cl, rl, 1795)
cloudRRR_L = clouds[0]
cloudRRR_R = clouds[1]
print len(cloudRRR_L), len(cloudRRR_R)
cloudRRR_L_csr = gen_csr(cloudRRR_L); cloudRRR_R_csr = gen_csr(cloudRRR_R)

mean_cloudRRR_L = cloudRRR_L_csr.mean(axis=0)
mean_dev_cloudRRR_L = get_meandeviation(cloudRRR_L_csr, 708, mean_cloudRRR_L)

print mean_dev_cloudRRR_L

mean_cloudRRR_R = cloudRRR_R_csr.mean(axis=0)
mean_dev_cloudRRR_R = get_meandeviation(cloudRRR_R_csr, 1087, mean_cloudRRR_R)

print mean_dev_cloudRRR_R

ans = [0] * 8580

for x in cloudleft_left:
    indl = x
    ind_final = cloudleft[indl]
    ans[ind_final] = 7

for x in cloudleft_right:
    indl = x
    ind_final = cloudleft[indl]
    ans[ind_final] = 6

for x in cloudRL_L:
    ind_cloudright_left = x
    ind_cloudright = cloudright_left[ind_cloudright_left]
    ind_final = cloudright[ind_cloudright]
    ans[ind_final] = 5

for x in cloudRL_R:
    ind_cloudright_left = x
    ind_cloudright = cloudright_left[ind_cloudright_left]
    ind_final = cloudright[ind_cloudright]
    ans[ind_final] = 4

for x in cloudRR_L:
    ind_cloudright_right = x
    ind_cloudright = cloudright_right[ind_cloudright_right]
    ind_final = cloudright[ind_cloudright]
    ans[ind_final] = 3

for x in cloudRRR_R:
    ind_cloudRR_R = x
    ind_cloudright_right = cloudRR_R[ind_cloudRR_R]
    ind_cloudright = cloudright_right[ind_cloudright_right]
    ind_final = cloudright[ind_cloudright]
    ans[ind_final] = 2

for x in cloudRRR_L:
    ind_cloudRR_R = x
    ind_cloudright_right = cloudRR_R[ind_cloudRR_R]
    ind_cloudright = cloudright_right[ind_cloudright_right]
    ind_final = cloudright[ind_cloudright]
    ans[ind_final] = 1

print ans

f2 = open('format2.dat', 'w')
for x in ans:
    f2.write(str(x))
    f2.write('\n')

f2.close()

x = [1,2,3,4,5,6,7]
y = []
y.append(mean_dev_cloudleft_left_csr)
y.append(mean_dev_cloudleft_right_csr)
y.append(mean_dev_cloudRL_L)
y.append(mean_dev_cloudRL_R)
y.append(mean_dev_cloudRR_L)
y.append(mean_dev_cloudRRR_R)
y.append(mean_dev_cloudRRR_L)

plt.bar(x,y,align='center')
plt.ylabel('Mean deviation')
plt.xlabel('Cluster numbers')

plt.show()
