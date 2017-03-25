import pandas as pd
import seaborn as sns
import matplotlib.image as image
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder

print 'import results...'
X_train = pd.read_csv('processed_data/test_data.csv')
Y_pred = pd.read_csv('processed_data/test_result.csv')
results = pd.concat([X_train, Y_pred], axis=1)
results['FRAME'] = results['WEEK'].astype(str) + "-" + results['DAY'].astype(str) + "-" + results['TIMESTEP'].astype(str)

#split result based on timefram based on the timesteps
g = results.drop(['WEEK', 'DAY', 'TIMESTEP'], axis=1)
g['FRAME'] = LabelEncoder().fit_transform(g.FRAME)

print 'create frame output...'
#create frame output
for t in range(600, 888):
    r = g[g.FRAME == t]
    
    FC = r.groupby(by=['X', 'Y'], as_index=False).GAP_FCST.sum()
    PRED = r.groupby(by=['X', 'Y'], as_index=False).GAP_PRED.sum()
    
    im = image.imread('porto.png')
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(16,5))
    ax1.set_title("PREDICTED GAP")
    ax2.set_title("ACTUAL GAP")

    ax1.imshow(im, extent=[0,10,0,7], aspect='auto')
    ax2.imshow(im, extent=[0,10,0,7], aspect='auto')
    
    FCS = FC.pivot('Y', 'X', 'GAP_FCST')
    PREDS = PRED.pivot('Y', 'X', 'GAP_PRED')
       
    f1=sns.heatmap(PREDS, ax=ax1, vmin=-10, vmax=10, cmap='seismic', alpha=0.6, robust=True)
    f2=sns.heatmap(FCS, ax=ax2, vmin=-10, vmax=10, cmap='seismic', alpha=0.6)
    
    f1.invert_yaxis()
    
    fig.savefig("output/file_" + str(t) + ".png")
    fig.clf()
    plt.close(fig)