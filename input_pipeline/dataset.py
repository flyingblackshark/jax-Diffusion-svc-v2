import grain.python
from input_pipeline import utils
import grain
from jax.sharding import Mesh
import jax
from input_pipeline import multihost_dataloading
from transformers import FlaxAutoModel
import glob
def get_dataset(hp,mesh):
    data_files = glob.glob(hp.data_loader.dataset_path)
    dataset = grain.python.ArrayRecordDataSource(data_files)
    index_sampler = grain.python.IndexSampler(
      num_records=len(dataset),
      #num_epochs=hp.data_loader.num_epochs,
      shard_options=grain.python.ShardOptions(
          shard_index=jax.process_index(), shard_count=hp.data_loader.host_number, drop_remainder=False
      ),
      shuffle=True,
      seed=hp.train.seed,
    )
    operations = []
    operations.append(utils.ParseFeatures(hp))
    operations.append(utils.SliceToLength(100))
    operations.append(grain.python.Batch(batch_size=hp.data_loader.global_batch_size // jax.process_count(), drop_remainder=False))
    dataloader = grain.python.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=index_sampler,
        worker_count=hp.data_loader.worker_count
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)
    return multihost_gen