# experimenting-with-Deep-Neural-Networks
An little set of experiments with deep neural network.

# AWS instance connection
To connect to the AWS instance run:
```bash
ssh -L localhost:8888:localhost:8888 -i chmod 0400 aws_key_ec2_gpu_tensorflow.pem  ubuntu@ec2-54-215-223-104.us-west-1.compute.amazonaws.com
```