from buffer import ReplayBuffer

buffer_size = 10
loop_size = 20

memory = ReplayBuffer(buffer_size, input_size=1,  n_actions=1)

for i in range(loop_size):
    memory.store_transition(i, i, i, i, i)

print("Testing to ensure the first state memory is correct")
assert memory.state_memory[0] == 10
print("Test passed\n")


print("Testing to ensure the first action memory is correct")
assert memory.action_memory[-1] == 19
print("Test passed\n")