package:
	python setup.py sdist
	python setup.py bdist_wheel

clean:
	rm -rf dist
	rm -rf build
	rm -rf bert_pytorch.egg-info
