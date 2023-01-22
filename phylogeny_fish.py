level2_mapping = [['Alosa chrysochloris'], # 9
                    ['Carassius auratus', 'Cyprinus carpio'] ,
                    ['Esox americanus'],
                    ['Gambusia affinis'],
                    ['Lepisosteus osseus', 'Lepisosteus platostomus'],
                    ['Lepomis auritus', 'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus', 'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 'Lepomis microlophus'],
                    ['Morone chrysops', 'Morone mississippiensis'],
                    ['Notropis atherinoides', 'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 'Notropis stramineus', 'Notropis telescopus', 'Notropis texanus', 'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis'],
                    ['Noturus exilis', 'Noturus flavus', 'Noturus gyrinus', 'Noturus miurus', 'Noturus nocturnus']]

level1_mapping = [['Alosa chrysochloris'], # 6
                    ['Carassius auratus', 'Cyprinus carpio', 'Notropis atherinoides', 'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 'Notropis stramineus', 'Notropis telescopus', 'Notropis texanus', 'Notropis volucellus', 'Notropis wickliffi', 'Phenacobius mirabilis'],
                    ['Esox americanus'],
                    ['Gambusia affinis', 'Lepomis auritus', 'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus', 'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 'Lepomis microlophus', 'Morone chrysops', 'Morone mississippiensis'],
                    ['Lepisosteus osseus', 'Lepisosteus platostomus'],
                    ['Noturus exilis', 'Noturus flavus', 'Noturus gyrinus', 'Noturus miurus', 'Noturus nocturnus']]

level0_mapping = [['Alosa chrysochloris', 'Carassius auratus', 'Cyprinus carpio', 'Notropis atherinoides', 'Notropis blennius', 'Notropis boops', 'Notropis buccatus', 'Notropis buchanani', 'Notropis dorsalis', 'Notropis hudsonius', 'Notropis leuciodus', 'Notropis nubilus', 'Notropis percobromus', 'Notropis stramineus', 'Notropis telescopus', 'Notropis texanus', 'Notropis volucellus', 'Notropis wickliffi', 'Noturus exilis', 'Noturus flavus', 'Noturus gyrinus', 'Noturus miurus', 'Noturus nocturnus', 'Phenacobius mirabilis'],
                    ['Esox americanus', 'Gambusia affinis', 'Lepomis auritus', 'Lepomis cyanellus', 'Lepomis gibbosus', 'Lepomis gulosus', 'Lepomis humilis', 'Lepomis macrochirus', 'Lepomis megalotis', 'Lepomis microlophus', 'Morone chrysops', 'Morone mississippiensis'],
                    ['Lepisosteus osseus', 'Lepisosteus platostomus']] # 3

all_species = sorted([x for y in level0_mapping for x in y])
species_to_idx = {all_species[i]:i for i in range(len(all_species))}

species_to_ances_level0 = {}
for i in range(len(level0_mapping)):
    for species in level0_mapping[i]:
        species_to_ances_level0[species_to_idx[species]] = i

species_to_ances_level1 = {}
for i in range(len(level1_mapping)):
    for species in level1_mapping[i]:
        species_to_ances_level1[species_to_idx[species]] = i

species_to_ances_level2 = {}
for i in range(len(level2_mapping)):
    for species in level2_mapping[i]:
        species_to_ances_level2[species_to_idx[species]] = i
