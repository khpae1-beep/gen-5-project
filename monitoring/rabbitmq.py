import pika
from django.conf import settings

PREDICTIONS_QUEUE = 'predictions'


def open_channel():
    """Connect to RabbitMQ at settings.RABBITMQ_HOST and declare the predictions queue.

    Returns (connection, channel); the caller closes the connection.
    """
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=settings.RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=PREDICTIONS_QUEUE, durable=True)
    return connection, channel
