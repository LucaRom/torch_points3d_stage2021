###### NORMAL LOADING FOR LAS DATASET
        # if self._split == "train":
        #     data_list = []
        #     for i in train_num:
        #         las_file = laspy.read(os.path.join(dataroot, "train", "{}.las".format(i)))
        #         # print(las_file)
        #
        #         las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)
        #
        #         las_label = np.array(las_file.classification).astype(np.int)
        #         # print(las_xyz)
        #         y = torch.from_numpy(las_label)
        #         # y = self._remap_labels(y)
        #         data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)
        #
        #         log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])
        #
        #         data_list.append(data)
        #
        #     data, slices = self.collate(data_list)
        #     torch.save((data, slices), self.processed_paths[0])
        #
        # elif self._split == "test":
        #     data_list = []
        #     for i in test_num:
        #         las_file = laspy.read(os.path.join(dataroot, "test", "{}.las".format(i)))
        #         # print(las_file)
        #
        #         las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)
        #
        #         las_label = np.array(las_file.classification).astype(np.int)
        #         # print(las_xyz)
        #         y = torch.from_numpy(las_label)
        #         # y = self._remap_labels(y)
        #         data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)
        #
        #         log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])
        #
        #         data_list.append(data)
        #
        #     data, slices = self.collate(data_list)
        #     torch.save((data, slices), self.processed_paths[1])
        #
        # else:
        #     raise ValueError("Split %s not recognised" % split)


#### GOOD FOR FONCTION BELOW
        # t1 = time.perf_counter() # Start time of processing (debugging)
        #
        # ### Create train dataset
        # pool = get_context("spawn").Pool()  # Creating pools for multiprocess                                            # Creating pools for multiprocess
        # create_samples_x = partial(create_subsamples, set_val="train")
        # data_list_temp = pool.map(create_samples_x, train_num)      # Mapping process with train_num range
        # flat_data_list = [x for z in data_list_temp for x in z]     # Flattening list since pool.map return list of lists
        #
        # # Create tensor and save it as "train.pt" wich is 'processed_paths[0]
        # data, slices = self.collate(flat_data_list)
        # torch.save((data, slices), self.processed_paths[0])
        #
        # pool.terminate()
        # pool.join()
        # pool.close()
        #
        # t2 = time.perf_counter() #end time of processing training data
        #
        # print(f'Processing training data finished in {t2-t1} seconds')
        #
        # ## Create test dataset
        # #Same as train dataset, but using 'test_num' for mapping and processed_paths[1] for 'test.pt'
        #
        #
        #
        # pool_test = get_context("spawn").Pool()  # Creating pools for multiprocess
        # create_samples_x = partial(create_subsamples, set_val="test")
        # data_list_temp = pool_test.map(create_samples_x, test_num)       # Mapping process with test_num range
        # flat_data_list = [x for z in data_list_temp for x in z]     # Flattening list since pool.map return list of lists
        #
        # # Create tensor and save it as "train.pt" wich is 'processed_paths[0]
        # data, slices = self.collate(flat_data_list)
        # torch.save((data, slices), self.processed_paths[1])
        #
        # pool.terminate()
        # pool.join()
        # pool.close()
