import tensorflow as tf
import tensorflow_probability as tfp

class ComputePosteriorNetwork():
    def __init__(self):

        # Create state to hold updated `step_size`.
        step_size = tf.get_variable(
            name='step_size',
            initializer=1.,
            use_resource=True,  # For TFE compatibility.
            trainable=False)

        # Initialize the HMC transition kernel.
        num_results = int(10e3)
        num_burnin_steps = int(1e3)
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=3,
            step_size=step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
                num_adaptation_steps=int(num_burnin_steps * 0.8)))

        # Run the chain (with burn-in).
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=1.,
            kernel=hmc)

        # Initialize all constructed variables.
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            init_op.run()
            samples_, kernel_results_ = sess.run([samples, kernel_results])

        print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
            samples_.mean(), samples_.std(), kernel_results_.is_accepted.mean()))
        # mean:-0.5003  stddev:0.7711  acceptance:0.6240