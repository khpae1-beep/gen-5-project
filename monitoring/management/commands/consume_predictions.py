import pika
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from monitoring.messages import InvalidMessage, handle_prediction_message
from monitoring.rabbitmq import PREDICTIONS_QUEUE, open_channel


class Command(BaseCommand):
    help = f'Read prediction messages from the {PREDICTIONS_QUEUE!r} RabbitMQ queue and save them as PredictionLog rows.'

    def handle(self, *args, **options):
        try:
            connection, channel = open_channel()
        except pika.exceptions.AMQPConnectionError as exc:
            raise CommandError(
                f'Could not connect to RabbitMQ at {settings.RABBITMQ_HOST!r}. Is it running (docker compose up -d)?'
            ) from exc

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=PREDICTIONS_QUEUE, on_message_callback=self.on_message)
        self.stdout.write(f'Waiting for messages on {PREDICTIONS_QUEUE!r}. Press Ctrl+C to stop.')
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            channel.stop_consuming()
        finally:
            connection.close()
        self.stdout.write('Stopped.')

    def on_message(self, channel, method, properties, body):
        try:
            log = handle_prediction_message(body)
        except InvalidMessage as exc:
            # A malformed message will never succeed, so drop it instead of requeueing.
            self.stderr.write(f'Discarding message: {exc}')
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        channel.basic_ack(delivery_tag=method.delivery_tag)
        self.stdout.write(f'Saved {log}')
