from projects.mnieeg import evaluation_collect

if __name__ == "__main__":
    print("Collecting data for channel evaluation")
    evaluation_collect.channel_collect()
    print("Collecting data for forward evaluation")
    evaluation_collect.forward_collect()
    evaluation_collect.forward_collect_distance_matrix()
    print("Collecting data for inverse evaluation")
    evaluation_collect.inverse_collect()

