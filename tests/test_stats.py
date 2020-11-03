import pytest

from milvus import DataType, BaseError, ParamError

from factorys import collection_name_factory


class TestGetCollectionStats:
    def test_get_collection_stats_normal(self, connect, vrecords):
        try:
            stats = connect.get_collection_stats(vrecords)
            assert stats["partition_count"] == 1
            assert stats["row_count"] == 10000
        except Exception as e:
            pytest.fail(f"Get collection {vrecords} stats fail: {str(e)}")

    def test_get_collection_stats_not_exist(self, connect):
        with pytest.raises(BaseError):
            collection_name = collection_name_factory()
            _ = connect.get_collection_stats(collection_name)

    @pytest.mark.parametrize("name", ["", 124, [1]])
    def test_get_collection_stats_invalid_name(self, name, connect):
        with pytest.raises(ParamError):
            _ = connect.get_collection_stats(name)
