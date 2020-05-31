def cluster_plot(df, file_path: str, label: str, prob: str, pred: str, cmap='gray'):
    num_clusters = len(df[label].unique())
    
    angle_list = [i * 2 * np.pi / num_clusters for i in range(num_clusters)]
    coord_list = [(np.cos(ang), np.sin(ang)) for ang in angle_list]

    for i, x in enumerate(coord_list):
        df.loc[df[pred] == i, 'coord'] = df.apply(lambda row: np.sum(np.array(coord_list) * np.array(row[prob]).reshape(-1, 1), axis=0), axis=1)
        
    group = list(df.groupby(pred))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    for i, (x, y) in enumerate(coord_list):
        try:
            img_avg = np.mean(np.stack([plt.imread(row[np.where(df.columns == file_path)[0][0]]) for _, row in group[i][1].iterrows()]), axis=0)
        except IndexError:
            img_avg = np.zeros_like(df.loc[0, file_path])

        thumbnail = AnnotationBbox(OffsetImage(img_avg, zoom=2, cmap=cmap), (x, y), frameon=False)
        ax.add_artist(thumbnail)

    for i, (x, y) in enumerate(coord_list):
        try:
            u, v  = list(zip(*df.loc[df[pred] == i, 'coord'].tolist()))
        except ValueError:
            pass

        plt.axis('off')
        plt.scatter(u, v, s=2, alpha=0.8)
    
    plt.show()