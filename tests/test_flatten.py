import torch


def test_flatten_unflatten():
    # Create two tensors that simulate images
    # Shape of each is (batch_size, C, W, H)
    # For simplicity use C, W only and set C to 4 and W to 5
    batch_size = 2
    C = 4
    W = 5
    grid_embeddings_A = torch.ones((batch_size, C, W)).cuda()
    # All entries of the first entry to a fixed value
    grid_embeddings_A[0, :] = 0
    grid_embeddings_A[1, :] = 1

    # And a second tensor to simulate the current grid embeddings
    grid_embeddings_B = torch.ones((batch_size, C, W)).cuda()
    # All entries of the first entry to a fixed value
    grid_embeddings_B[0, :] = 2
    grid_embeddings_B[1, :] = 3

    mem = torch.cuda.memory_allocated()
    # Step 1: Stack them together
    pillar_embeddings = torch.stack((grid_embeddings_A, grid_embeddings_B), dim=1)
    print(f"Step 1 took {(torch.cuda.memory_allocated() - mem)} B GPU memory")
    # Size should be (batch_size, 2, C, W)
    assert pillar_embeddings.size() == (batch_size, 2, C, W)

    # First entry should be (2, C, W) with [0] being all zeros and [1] being all 2
    assert pillar_embeddings[0][0].allclose(torch.tensor(0, dtype=torch.float32))
    assert pillar_embeddings[0][1].allclose(torch.tensor(2, dtype=torch.float32))
    # Analog for second
    assert pillar_embeddings[1][0].allclose(torch.tensor(1, dtype=torch.float32))
    assert pillar_embeddings[1][1].allclose(torch.tensor(3, dtype=torch.float32))

    # Step 2: Flatten the first two dimensions to act as a larger batch
    mem = torch.cuda.memory_allocated()
    pillar_embeddings_flattened = pillar_embeddings.flatten(0, 1)
    print(f"Step 2 took {(torch.cuda.memory_allocated() - mem)} B GPU memory")
    # Should be (batch_size * 2, C, W)
    assert pillar_embeddings_flattened.size() == (batch_size * 2, C, W)

    # Same as above but without second dimensions
    assert pillar_embeddings_flattened[0].allclose(torch.tensor(0, dtype=torch.float32))
    assert pillar_embeddings_flattened[1].allclose(torch.tensor(2, dtype=torch.float32))
    # Analog for second
    assert pillar_embeddings_flattened[2].allclose(torch.tensor(1, dtype=torch.float32))
    assert pillar_embeddings_flattened[3].allclose(torch.tensor(3, dtype=torch.float32))

    # Step 3: Now the (prev, cur) dimension should be reconstructed such that later computations can be made
    mem = torch.cuda.memory_allocated()
    pillar_embeddings_unflattened = pillar_embeddings_flattened.unflatten(0, (batch_size, 2))
    print(f"Step 3 took {(torch.cuda.memory_allocated() - mem)} B GPU memory")
    # Should be the same as above
    # First entry should be (2, C, W) with [0] being all zeros and [1] being all 2
    assert pillar_embeddings[0][0].allclose(torch.tensor(0, dtype=torch.float32))
    assert pillar_embeddings[0][1].allclose(torch.tensor(2, dtype=torch.float32))
    # Analog for second
    assert pillar_embeddings[1][0].allclose(torch.tensor(1, dtype=torch.float32))
    assert pillar_embeddings[1][1].allclose(torch.tensor(3, dtype=torch.float32))

    # Good so far
    # Step 4: Now the second dimension is flattened which concatenates the (prev, cur) at channel dimension
    mem = torch.cuda.memory_allocated()
    pillar_embeddings_depth = pillar_embeddings_unflattened.flatten(1, 2)
    print(f"Step 4 took {(torch.cuda.memory_allocated() - mem)} B GPU memory")
    # Shape should be (batch_size, 2 * C, W)
    assert pillar_embeddings_depth.size() == (batch_size, 2 * C, W)
    # First batch entry should only have 0 and 2, second batch entry should only have 1 and 3
    assert pillar_embeddings_depth[0].unique(sorted=True).allclose(torch.tensor([0, 2], dtype=torch.float32,
                                                                                device="cuda"))
    assert pillar_embeddings_depth[1].unique(sorted=True).allclose(torch.tensor([1, 3], dtype=torch.float32,
                                                                                device="cuda"))

    # And thats it


if __name__ == '__main__':
    test_flatten_unflatten()
