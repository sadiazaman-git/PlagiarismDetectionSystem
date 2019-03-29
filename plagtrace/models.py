from __future__ import unicode_literals
from django.db import models

class Contact(models.Model):
    username = models.TextField(max_length=200, default= None)
    password = models.CharField(max_length=100 , default=None)
    subject = models.CharField(max_length=500)
    description = models.TextField(max_length=2000)

    def __str__(self):
        return self.username

class Document(models.Model):
    file1 = models.FileField()
    file2 = models.FileField()

    def __str__(self):
        return self.file1.name + ' ' + self.file2.name

