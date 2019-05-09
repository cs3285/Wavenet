t1 = variables['postprocessing']['postprocess1'].eval(session=sess)
t2 = variables['postprocessing']['postprocess1'].eval(session=sess)

t3 = variables['postprocessing']['postprocess1'].eval(session=sess)

t4 = variables['postprocessing']['postprocess1'].eval(session=sess)


from matplotlib import pyplot as plt

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)





r_idx = sample(range(len(df_train) - 7200),1)[0]

r_idx+=10
ts = np.copy(df_train[r_idx:r_idx+7400,:].reshape([1,7400, -1]))
ts_predict = np.copy(df_train[r_idx:r_idx+7400,:].reshape([1,7400, -1]))


for i in range(200):
    ipt = np.copy(ts_predict[:,i:i+7200,:])
    ro = sess.run(raw_output, feed_dict={input_batch: ipt})
    ts_predict[0,7199+i,:] = ro[0,-1,:]



fig, ax = plt.subplots(nrows=3, ncols=3)
for i in range(8):
    ax1 = ax[i//3][i%3]
    tsc1 = [1]
    for j in range(200):
        tsc1.append(tsc1[-1] * np.exp(ts[0,j+7199,i]))
    tsc2 = [1]
    for j in range(200):
        tsc2.append(tsc2[-1] * np.exp(5*ts_predict[0,j+7199,i]))  
    ax1.plot(tsc1, color='tab:red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax1.plot(tsc2, color='tab:blue')
    #align_yaxis(ax1, 0, ax2, 0)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show(block=False)